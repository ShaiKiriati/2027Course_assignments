#!/usr/bin/env python3
"""
Hidden-object game built with SAM 3.

SAM 3 segments by *concept*: you give it a short noun phrase ("dog", "coffee
mug") and it returns a mask for every matching instance, already named. The
program runs SAM 3 over a list of candidate concepts, picks 5-20 of the objects
it finds at random, and shows the image with their names listed below.

Play: left-click an object in the picture. Its name is struck off the list and
the object is covered by a dark overlay. Found objects stay struck off and
darkened until you have found them all.
Press N for a new game (a fresh random set from the same image; no re-run of SAM 3).
Concept group words are stored in clip_word_groups.json next to this file.

Setup:
    pip install -U transformers torch pillow numpy matplotlib accelerate
    # facebook/sam3 is a gated model: request access at
    # https://huggingface.co/facebook/sam3 , then log in:
    hf auth login

Run:
    python segment_and_name_sam3.py                    # file dialog
    python segment_and_name_sam3.py photo.jpg
    python segment_and_name_sam3.py photo.jpg --concepts my_concepts.txt
    python segment_and_name_sam3.py photo.jpg                      # auto-picks concept groups from the scene
    python segment_and_name_sam3.py photo.jpg --groups bathroom,toys   # choose groups yourself
    python segment_and_name_sam3.py photo.jpg --fast                # 560px: much faster on CPU
    python segment_and_name_sam3.py photo.jpg --pool 40            # stop after 40 distinct objects (0 = search all concepts)
    # Results are cached per photo, so the second run starts instantly (--recompute to redo).
    python segment_and_name_sam3.py photo.jpg --groups refrigerator --half   # fastest settings
    python segment_and_name_sam3.py photo.jpg --cheat  # tint all hidden objects (debugging)
"""
import os

# Windows without Developer Mode cannot make symlinks; Hugging Face then stores plain copies,
# which works fine. Silence the warning about it (must be set before the libraries load).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import hashlib
import json
import random
import threading
import time
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor, Sam3Config, Sam3Model, Sam3Processor

SAM3_MODEL = "facebook/sam3"
DEFAULT_POOL = 30       # stop searching once this many distinct objects are found (0 = search everything)
MIN_POOL = 20           # a game uses up to 20 objects, so never stop below this
DEFAULT_MAX_PER_CONCEPT = 3   # keep at most this many instances per concept (best scores)
FAST_SIZE = 560        # resolution used by --fast (SAM 3 normally runs at 1008)
CACHE_VERSION = 1      # bump to invalidate old caches
MAX_SIDE = 1280        # downscale very large images for speed/memory
MIN_AREA = 0.001       # ignore instances smaller than this fraction of the image
DEDUPE_IOU = 0.7       # drop overlapping instances found under different concepts

CONCEPT_GROUPS_PATH = Path(__file__).with_name("clip_word_groups.json")
_CONCEPT_GROUPS_LOCK = threading.Lock()


def load_concept_groups():
    try:
        groups = json.loads(CONCEPT_GROUPS_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not read CLIP word groups at {CONCEPT_GROUPS_PATH}: {e}") from e
    if not isinstance(groups, dict) or not groups:
        raise RuntimeError(f"Invalid CLIP word groups file: {CONCEPT_GROUPS_PATH}")
    for group, words in groups.items():
        if (not isinstance(group, str) or not group.strip() or not isinstance(words, list)
                or any(not isinstance(word, str) or not word.strip() for word in words)):
            raise RuntimeError(f"Invalid CLIP word groups file: {CONCEPT_GROUPS_PATH}")
    return groups


CONCEPT_GROUPS = load_concept_groups()

# Used by --groups auto: CLIP compares the whole image against these descriptions
# to decide which concept groups are worth searching. "general" is always included.
SCENE_MODEL = "openai/clip-vit-base-patch32"   # small and fast; enough for scene type
AUTO_TOP_GROUPS = 3
GROUP_SCENE_PROMPTS = {
    "closet": ["a photo of a closet", "a photo of clothes hanging in a wardrobe",
               "a photo of shoes and clothes on closet shelves"],
    "dresser": ["a photo of a dresser with open drawers",
                "a photo of folded clothes and jewelry on a dresser"],
    "refrigerator": ["a photo of the inside of a refrigerator",
                     "a photo of food on refrigerator shelves"],
    "dishwasher": ["a photo of the inside of a dishwasher",
                   "a photo of clean dishes and glasses in a dish rack"],
    "pantry": ["a photo of a kitchen pantry",
               "a photo of a shelf with canned food and jars"],
    "suitcase": ["a photo of an open suitcase",
                 "a photo of packed luggage with travel items"],
    "desk": ["a photo of an office desk",
             "a photo of a desk with stationery and a computer"],
    "toys": ["a photo of a pile of toys", "a photo of a children's playroom"],
    "games": ["a photo of board games and playing cards",
              "a photo of video games and game consoles"],
    "bathroom": ["a photo of a bathroom", "a photo of a bathroom sink with toiletries"],
    "bedroom": ["a photo of a bedroom", "a photo of a bed with pillows and blankets"],
}


class Cancelled(Exception):
    """Raised when the user cancels a search from the GUI."""


@dataclass
class Instance:
    name: str
    mask: np.ndarray   # bool, H x W
    score: float


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Open an image",
        filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def load_concepts(path, groups="all"):
    if path:
        with open(path, encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
    else:
        items, _sources = load_concepts_with_groups(groups)
        return items
    seen, out = set(), []
    for c in items:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_concepts_with_groups(groups="all"):
    concept_groups = load_concept_groups()
    names = list(concept_groups) if groups == "all" else [g.strip() for g in groups.split(",")]
    unknown = [g for g in names if g not in concept_groups]
    if unknown:
        raise SystemExit(f"Unknown group(s): {unknown}. Choose from: {list(concept_groups)}")
    items, sources, indices = [], [], {}
    for group in names:
        for concept in concept_groups[group]:
            key = concept.casefold()
            if key not in indices:
                indices[key] = len(items)
                items.append(concept)
                sources.append([group])
            elif group not in sources[indices[key]]:
                sources[indices[key]].append(group)
    return items, sources


def save_concept_word(group, word):
    if not isinstance(group, str):
        raise ValueError("Choose a valid word group.")
    if not isinstance(word, str):
        raise ValueError("Enter a word or short phrase.")
    normalized = " ".join(word.split())
    if (not normalized or len(normalized) > 80 or len(normalized.split()) > 5
            or any(ord(char) < 32 or ord(char) == 127 for char in normalized)):
        raise ValueError("Use a non-empty label of up to five words and 80 characters.")
    with _CONCEPT_GROUPS_LOCK:
        groups = load_concept_groups()
        if group not in groups:
            raise ValueError(f"Unknown word group: {group}")
        if any(existing.casefold() == normalized.casefold() for existing in groups[group]):
            return normalized, False
        groups[group].append(normalized)
        temp_path = CONCEPT_GROUPS_PATH.with_suffix(CONCEPT_GROUPS_PATH.suffix + ".tmp")
        temp_path.write_text(json.dumps(groups, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(CONCEPT_GROUPS_PATH)
    return normalized, True


@torch.no_grad()
def detect_groups(image, device, top_k, clip_name):
    """Pick the most relevant concept groups for this image using CLIP scene recognition.

    One CLIP pass over the whole image, compared against a few descriptions per group
    (e.g. "a photo of a bathroom"). Much cheaper than searching every group with SAM 3.
    """
    print(f"Recognising the scene with {clip_name}...")
    model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_name)

    names = list(GROUP_SCENE_PROMPTS)
    prompts, owner = [], []
    for g in names:
        for text in GROUP_SCENE_PROMPTS[g]:
            prompts.append(text)
            owner.append(g)

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    logits = model(**inputs).logits_per_image[0].detach().float().cpu().numpy()  # one per prompt

    # Score each group as the mean logit of its prompts, then softmax across groups.
    scores = np.array([np.mean([l for l, o in zip(logits, owner) if o == g]) for g in names])
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()

    order = np.argsort(-probs)
    picked = [names[i] for i in order[:top_k]]
    print("  scene scores: " + ", ".join(f"{names[i]} {probs[i]:.2f}" for i in order[:6]))

    del model, processor  # free memory before SAM 3 loads
    if device == "cuda":
        torch.cuda.empty_cache()
    return picked


def load_model(device, half, fast):
    """Load SAM 3. half = half precision (CUDA only); fast = lower input resolution."""
    kwargs = {}
    if fast:
        config = Sam3Config.from_pretrained(SAM3_MODEL)
        config.image_size = FAST_SIZE
        kwargs["config"] = config
    if half and device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using {dtype} precision")
        try:
            model = Sam3Model.from_pretrained(SAM3_MODEL, dtype=dtype, **kwargs)
        except TypeError:   # older transformers call this argument torch_dtype
            model = Sam3Model.from_pretrained(SAM3_MODEL, torch_dtype=dtype, **kwargs)
    else:
        if half:
            print("--half only applies on CUDA GPUs; using full precision")
        model = Sam3Model.from_pretrained(SAM3_MODEL, **kwargs)
    return model.to(device).eval()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def find_instances(image, concepts, device, threshold, half, fast=False,
                   pool=DEFAULT_POOL, max_per_concept=DEFAULT_MAX_PER_CONCEPT, seed=None,
                   stop_event=None):
    """Run SAM 3 over the concepts (in random order) and collect named instances.

    Concepts are shuffled and searching stops once `pool` distinct objects have been
    found, so a photo with plenty of objects needs only part of the list while the
    pool is still a random sample. pool=0 searches the whole list.

    The expensive vision backbone runs ONCE for the image; each concept then only
    needs the (much lighter) text + detection + mask heads, reusing those embeddings.
    """
    print(f"Loading {SAM3_MODEL} on {device}...")
    if device == "cpu":
        print("  Warning: running on CPU. SAM 3 is very slow here; a CUDA GPU is strongly recommended.")
        print(f"  PyTorch {torch.__version__}; built with CUDA: {torch.version.cuda or 'NO (CPU-only build)'}; "
              f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        print("  With an NVIDIA GPU, reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/")
        print("  Tips for CPU: use --fast, fewer concepts (--auto-top 1 or a short --concepts file). "
              "Results are cached, so each photo only needs one slow run.")
    if fast:
        print(f"Fast mode: running SAM 3 at {FAST_SIZE}px (quicker, somewhat less accurate)")
    model = load_model(device, half, fast)
    proc_kwargs = {"size": {"height": FAST_SIZE, "width": FAST_SIZE}} if fast else {}
    processor = Sam3Processor.from_pretrained(SAM3_MODEL, **proc_kwargs)

    t0 = time.time()
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values.to(model.dtype))
    sync(device)
    print(f"Image encoded once in {time.time() - t0:.1f}s")
    sizes = img_inputs.get("original_sizes").tolist()

    order = list(concepts)
    random.Random(seed).shuffle(order)   # random order => an early stop still gives a random sample
    found = []
    t1 = time.time()
    searched = 0
    for name in order:
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()
        searched += 1
        text_inputs = processor(text=name, return_tensors="pt").to(device)
        outputs = model(vision_embeds=vision_embeds, **text_inputs)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=sizes,
        )[0]
        # Best-scoring instances first; cap per concept so one concept can't fill the pool.
        hits = sorted(zip(res["masks"], (float(s) for s in res["scores"])),
                      key=lambda t: t[1], reverse=True)[:max_per_concept]
        for mask, score in hits:
            m = mask.detach().cpu().numpy().astype(bool)
            if m.mean() >= MIN_AREA:
                found.append(Instance(name, m, score))

        if searched % 5 == 0 or searched == len(order):
            found = dedupe(found)   # count distinct objects, not overlapping duplicates
            per = (time.time() - t1) / searched
            target = f" (stopping at {pool})" if pool else ""
            print(f"  searched {searched}/{len(order)} concepts, {len(found)} distinct objects{target}, "
                  f"{per:.2f}s/concept", end="\r")
            if pool and len(found) >= pool:
                print(f"\n  Enough objects found after {searched} of {len(order)} concepts; stopping early.")
                break
    else:
        print()
    return found


def dedupe(instances):
    """Keep the highest-scoring instance when several concepts hit the same object
    (e.g. 'sofa' and 'couch', or 'person' and 'woman')."""
    kept, small = [], []
    for inst in sorted(instances, key=lambda i: i.score, reverse=True):
        s = inst.mask[::4, ::4]   # downsample for a fast IoU check
        dup = False
        for k_small in small:
            inter = np.logical_and(s, k_small).sum()
            union = np.logical_or(s, k_small).sum()
            if union and inter / union > DEDUPE_IOU:
                dup = True
                break
        if not dup:
            kept.append(inst)
            small.append(s)
    return kept


FOUND_TEXT_COLOR = "#8a8a8a"       # greyed-out colour for names already found
DARKEN_RGBA = (0, 0, 0, 175)       # dark overlay laid over found objects (alpha 0-255)
ROWS_PER_COLUMN = 7


def strike(text):
    """Strike-through via the Unicode combining long stroke (matplotlib has no strike style)."""
    return "".join(ch + "\u0336" for ch in text)


def unique_labels(chosen):
    """'person' appearing twice becomes 'person #1' and 'person #2'."""
    totals = Counter(i.name for i in chosen)
    seen = Counter()
    labels = []
    for inst in chosen:
        if totals[inst.name] > 1:
            seen[inst.name] += 1
            labels.append(f"{inst.name} #{seen[inst.name]}")
        else:
            labels.append(inst.name)
    return labels


def pick_chosen(instances, namer=None, stop_event=None):
    """Pick 5-20 objects at random (fewer if the image doesn't have that many).

    With a `namer`, the instances are unnamed segments (name None): only the picked
    ones get named, and any the namer rejects (name "") are replaced by other random ones.
    `stop_event` (optional) lets a GUI cancel the naming between and during batches.
    """
    target = random.randint(5, 20)
    if namer is None:
        n = min(target, len(instances))
        return random.sample(instances, n) if n else []
    pool = [i for i in instances if i.name != ""]     # "" = already rejected
    random.shuffle(pool)
    chosen = []
    while pool and len(chosen) < target:
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()
        batch = [pool.pop() for _ in range(min(target - len(chosen), len(pool)))]
        namer(batch, stop_event)
        chosen += [i for i in batch if i.name]
    return chosen


def show(image, instances, overlay, namer=None, first_game=None):
    """Hidden-object game: find the listed objects in the picture.

    Left-click an object: its name is struck off the list and the object is
    covered with a dark overlay. Found objects stay struck off and darkened.
    Press N for a new game: a fresh random set of objects from the same image
    (SAM 3 is not run again).

    namer / first_game are used by the free-form version (segment_then_name.py): the
    instances are unnamed segments, `namer(list_of_instances)` names them on demand,
    and `first_game` is the already-named set for the opening game.
    """
    original = np.asarray(image).astype(np.float32)
    h, w = original.shape[:2]

    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.3])

    ax_img = fig.add_subplot(gs[0])
    ax_img.axis("off")
    img_artist = ax_img.imshow(original.astype(np.uint8))
    # Transparent layer on top of the picture; found objects get painted dark into it.
    dark = np.zeros((h, w, 4), dtype=np.uint8)
    dark_layer = ax_img.imshow(dark, interpolation="nearest")

    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")
    title = ax_txt.text(0.5, 1.0, "", ha="center", va="top", fontsize=12, transform=ax_txt.transAxes)

    if not instances:
        title.set_text("No objects found. Try lowering --threshold or supplying more --concepts.")
        fig.tight_layout()
        plt.show()
        return

    ax_txt.text(0.5, 0.0, "Press N for a new game", ha="center", va="bottom",
                fontsize=9, color="#888888", transform=ax_txt.transAxes)

    default_color = plt.rcParams["text.color"]
    game = {"chosen": [], "labels": [], "texts": [], "areas": [], "found": set()}

    def render_base(chosen):
        base = original.copy()
        if overlay:  # cheat/debug mode: tint every hidden object
            cmap = plt.get_cmap("tab20")
            for i, inst in enumerate(chosen):
                if inst.mask.shape == (h, w):
                    color = np.array(cmap(i % 20)[:3]) * 255
                    base[inst.mask] = base[inst.mask] * 0.6 + color * 0.4
        return base.astype(np.uint8)

    def update_title():
        total, n_found = len(game["chosen"]), len(game["found"])
        if n_found == total:
            title.set_text(f"All {total} objects found!  Press N for a new game")
            title.set_fontweight("bold")
        else:
            title.set_text(f"Found {n_found} of {total}: click the objects in the picture")
            title.set_fontweight("normal")

    pending_first = [first_game] if first_game else []

    def new_game():
        if pending_first:
            chosen = pending_first.pop()   # named earlier, while the photo was being prepared
        else:
            if namer is not None:
                title.set_text("Choosing and naming objects\u2026 this can take a while")
                title.set_fontweight("normal")
                fig.canvas.draw()
                fig.canvas.flush_events()
            chosen = pick_chosen(instances, namer)
        for t in game["texts"]:
            t.remove()
        if not chosen:
            game.update(chosen=[], labels=[], texts=[], areas=[], found=set())
            dark[:] = 0
            dark_layer.set_data(dark)
            title.set_text("No nameable objects found. Press N to try again")
            fig.canvas.draw_idle()
            return
        labels = unique_labels(chosen)
        rows = min(ROWS_PER_COLUMN, len(labels))
        ncols = math.ceil(len(labels) / rows)
        col_w = 0.3
        left = 0.5 - ncols * col_w / 2
        texts = []
        for i, label in enumerate(labels):
            col, row = divmod(i, rows)
            texts.append(ax_txt.text(
                left + col * col_w, 0.82 - row * 0.115, f"\u2022 {label}",
                ha="left", va="top", fontsize=11, color=default_color,
                transform=ax_txt.transAxes,
            ))
        game.update(chosen=chosen, labels=labels, texts=texts,
                    areas=[inst.mask.sum() for inst in chosen], found=set())
        dark[:] = 0
        dark_layer.set_data(dark)
        img_artist.set_data(render_base(chosen))
        update_title()
        print("Chosen:", [i.name for i in chosen])
        fig.canvas.draw_idle()

    def mark_found(idx):
        game["found"].add(idx)
        dark[game["chosen"][idx].mask] = DARKEN_RGBA
        dark_layer.set_data(dark)
        game["texts"][idx].set_text(f"\u2022 {strike(game['labels'][idx])}")
        game["texts"][idx].set_color(FOUND_TEXT_COLOR)
        update_title()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax_img or event.xdata is None:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan tool is active; don't treat the click as a guess
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if not (0 <= x < w and 0 <= y < h):
            return
        # If several objects overlap at this pixel, the smallest (most specific) one is
        # the target. Clicking empty space or an already-found object does nothing, so
        # a stray second click on a found item never reveals the larger object behind it.
        hits = [i for i, inst in enumerate(game["chosen"]) if inst.mask[y, x]]
        if hits:
            target = min(hits, key=lambda i: game["areas"][i])
            if target not in game["found"]:
                mark_found(target)

    def on_key(event):
        if event.key in ("n", "N"):
            new_game()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    new_game()
    fig.tight_layout()
    plt.show()


def cache_path_for(cache_dir, image_path, args):
    """Cache file name derived from the image *content* and every setting that changes the result."""
    with open(image_path, "rb") as f:
        image_hash = hashlib.sha1(f.read()).hexdigest()
    concepts_hash = None
    if args.concepts:
        with open(args.concepts, encoding="utf-8") as f:
            concepts_hash = hashlib.sha1(f.read().encode("utf-8")).hexdigest()
    settings = {
        "version": CACHE_VERSION, "image": image_hash, "model": SAM3_MODEL,
        "fast": bool(args.fast), "threshold": args.threshold, "max_side": MAX_SIDE,
        "min_area": MIN_AREA, "dedupe_iou": DEDUPE_IOU,
        "concepts_file": concepts_hash,
        "pool": args.pool, "seed": args.seed, "max_per_concept": args.max_per_concept,
        "groups": None if args.concepts else args.groups,
        "auto_top": args.auto_top if (args.groups == "auto" and not args.concepts) else None,
    }
    key = hashlib.sha1(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{key}.sam3cache.npz"   # content-based: works for copies/renames of the photo


def save_cache(path, instances, shape):
    h, w = shape
    packed = [np.packbits(i.mask.reshape(-1)) for i in instances]
    masks = np.stack(packed) if packed else np.zeros((0, (h * w + 7) // 8), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, names=np.array([i.name for i in instances], dtype=str),
        scores=np.array([i.score for i in instances], dtype=np.float32),
        masks=masks, shape=np.array([h, w]),
    )


def load_cache(path, shape):
    """Return the cached instances, or None if the file is unreadable or doesn't fit this image."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if tuple(int(x) for x in data["shape"]) != tuple(shape):
                return None
            h, w = shape
            return [
                Instance(str(name), np.unpackbits(row)[:h * w].reshape(h, w).astype(bool), float(score))
                for name, score, row in zip(data["names"], data["scores"], data["masks"])
            ]
    except Exception as e:   # corrupt or incompatible cache: just recompute
        print(f"  (ignoring unreadable cache: {e})")
        return None


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="PNG/JPEG path (omit for a file dialog)")
    p.add_argument("--concepts", help="text file with one concept phrase per line (overrides --groups)")
    p.add_argument(
        "--groups", default="auto",
        help="'auto' (default): CLIP recognises the scene and picks the most relevant groups; "
             "'all'; or a comma-separated list from general,closet,dresser,refrigerator,dishwasher,"
             "pantry,suitcase,desk,toys,games,bathroom,bedroom",
    )
    p.add_argument("--auto-top", type=int, default=AUTO_TOP_GROUPS,
                   help="with --groups auto: how many groups to search besides 'general' (more = slower, fewer misses)")
    p.add_argument("--scene-model", default=SCENE_MODEL, help="CLIP model used by --groups auto")
    p.add_argument("--threshold", type=float, default=0.5, help="SAM 3 detection threshold")
    p.add_argument("--half", action="store_true",
                   help="run in half precision on a CUDA GPU (faster, less memory)")
    p.add_argument("--pool", type=int, default=DEFAULT_POOL,
                   help=f"search concepts in random order and stop once this many distinct objects are found "
                        f"(default {DEFAULT_POOL}; minimum {MIN_POOL}; 0 = search every concept)")
    p.add_argument("--seed", type=int, help="seed for the random concept order (reproducible searches)")
    p.add_argument("--max-per-concept", type=int, default=DEFAULT_MAX_PER_CONCEPT,
                   help="keep at most this many instances of any one concept")
    p.add_argument("--fast", action="store_true",
                   help=f"run SAM 3 at {FAST_SIZE}px instead of 1008: much faster on CPU, somewhat less accurate")
    p.add_argument("--no-cache", action="store_true", help="don't read or write the result cache")
    p.add_argument("--recompute", action="store_true", help="ignore any cached result and overwrite it")
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "sam3_hidden_objects"),
                   help="where results are cached (copy a cache file to another machine to reuse it)")
    p.add_argument("--batch", type=int, help=argparse.SUPPRESS)  # no longer used; kept so old commands still run
    p.add_argument("--overlay", "--cheat", dest="overlay", action="store_true",
                   help="cheat mode: tint all hidden objects so you can see where they are")
    return p


def prepare(path, args, stop_event=None):
    """Load the photo and find its objects (from the cache when possible).

    Returns (image, instances, extras); extras are extra keyword arguments for show().
    Used by both the command line and the GUI launcher.
    """
    def check_cancel():
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phone photos: apply the rotation tag
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    shape = (image.height, image.width)
    if 0 < args.pool < MIN_POOL:
        print(f"Pool size raised to {MIN_POOL}, since a game uses up to 20 objects")
        args.pool = MIN_POOL

    cache_file = None if args.no_cache else cache_path_for(args.cache_dir, path, args)
    instances = None
    if cache_file and cache_file.exists() and not args.recompute:
        instances = load_cache(cache_file, shape)
        if instances is not None:
            print(f"Loaded {len(instances)} objects from cache, skipping SAM 3 ({cache_file})")

    if instances is None:
        device = pick_device()
        groups = args.groups
        if groups == "auto" and not args.concepts:
            picked = detect_groups(image, device, args.auto_top, args.scene_model)
            groups = ",".join(["general"] + picked)
            print(f"Auto-selected groups: {groups}")
        check_cancel()
        concepts = load_concepts(args.concepts, groups)
        print(f"Searching for {len(concepts)} concepts")
        instances = find_instances(image, concepts, device, args.threshold, args.half, args.fast,
                                   args.pool, args.max_per_concept, args.seed, stop_event)
        instances = dedupe(instances)
        print(f"{len(instances)} distinct objects found")
        if cache_file:
            save_cache(cache_file, instances, shape)
            print(f"Saved results to cache: {cache_file}")
    return image, instances, {}


def main():
    args = build_parser().parse_args()
    path = args.image or choose_file()
    if not path:
        print("No file selected.")
        return
    image, instances, extras = prepare(path, args)
    show(image, instances, args.overlay, **extras)


if __name__ == "__main__":
    main()
