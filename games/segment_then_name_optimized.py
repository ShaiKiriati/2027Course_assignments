#!/usr/bin/env python3
"""
Hidden Object Game, open-ended version: segment first, name afterwards.

1. A vision-language model tags the entire photo with a concise scene description.
2. SAM (SAM 3 by default) segments everything it can find in the photo, with no
   word list involved.
3. Each game picks 5-20 of those segments at random and names only those, using
   the whole-image description as context:
   - "vlm" (default): a small vision-language model looks at each segment and says
     what it is, in its own words (open-ended names).
   - "clip": CLIP picks the best match from the word groups (much faster).
4. The names become the list in the game window. Click objects to find them.
   Press N for a new game: new random segments are picked and named.

Keep this file next to segment_and_name_sam3.py (it reuses the game window).

Setup (same environment as the other version):
    pip install -U transformers torch pillow numpy matplotlib accelerate

Run:
    python segment_then_name.py photo.jpg
    python segment_then_name.py photo.jpg --namer clip --groups bathroom,toys
    python segment_then_name.py photo.jpg --sam-model facebook/sam2.1-hiera-small   # no gated model, faster
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor, CLIPModel,
                          CLIPProcessor, pipeline)

import segment_and_name_sam3 as base
from segment_and_name_sam3 import (CONCEPT_GROUPS, Cancelled, Instance, choose_file,  # noqa: F401
                                   detect_groups, load_concepts, pick_device, show)

SAM_MODEL = "facebook/sam2.1-hiera-small"       # smaller, ungated default for CPU-friendly use
VLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"  # open-ended naming (500M version: better, slower)
CLIP_MODEL = "openai/clip-vit-base-patch32"        # lighter CLIP model for CPU use
DEFAULT_GRID = 12       # SAM prompts a grid of GRID x GRID points; more = finds smaller things, slower
DEFAULT_MAX_SIDE = 768  # reduce SAM/VLM work on CPU; increase for small-object detail
MAX_SEGMENTS = 80       # keep the best-scoring segments only
MIN_AREA, MAX_AREA = 0.004, 0.60   # allowed segment size (fraction of the image)
NMS_IOU = 0.6           # drop near-duplicate segments
CLIP_MIN_CONF = 0.15    # --namer clip: reject names CLIP is unsure about
VLM_CROP_SIZE = 256     # smaller inputs reduce CPU inference time and memory
CACHE_VERSION = 3
REJECT_LABELS = ["background", "blurry background", "a small part of something", "a shadow", "a texture"]
SCENE_PROMPT = ("Describe the entire image in one concise sentence. Identify the overall scene, setting, "
                "and the main kinds of objects visible. Be factual and do not guess details.")
VLM_PROMPT = ("The full image shows this scene: {scene_context}\n"
              "This crop is one segmented region from that image. Identify the specific object in the crop, "
              "using the scene description only as context. Do not name the whole scene. "
              "Answer with the object's name only, in one to four words. "
              "If it is only background, a texture or an unidentifiable fragment, answer: unclear")

BAD_NAMES = {"unclear", "unknown", "none", "nothing", "background", "image", "picture", "photo",
             "texture", "pattern", "n/a", "na", "object", "item", "thing", "something"}
LEADING = re.compile(r"^(the|a|an|this is|it is|it's|there is|there's|image of|picture of|photo of|close up of)\s+")


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def segment(image, sam_model, grid, device, points_per_batch):
    """Automatic mask generation (SAM prompts a grid of points), then filter and de-duplicate."""
    print(f"Segmenting with {sam_model} on {device} ({grid}x{grid} points)...")
    if device == "cpu":
        print("  On a CPU this step can take a few minutes. It only runs once per photo (results are cached).")
    generator = pipeline("mask-generation", model=sam_model, device=device)
    try:
        out = generator(image, points_per_batch=points_per_batch, points_per_crop=grid)
    except (TypeError, ValueError):   # a transformers version without the grid option
        out = generator(image, points_per_batch=points_per_batch)
    masks = [torch.as_tensor(m).cpu().numpy().astype(bool) for m in out["masks"]]
    scores = [float(s) for s in out["scores"]]

    total = masks[0].size if masks else 1
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept, small_kept = [], []
    for i in order:
        if not (MIN_AREA <= masks[i].sum() / total <= MAX_AREA):
            continue
        small = masks[i][::4, ::4]   # downsample for a fast overlap check
        if any(mask_iou(small, k) > NMS_IOU for k in small_kept):
            continue
        kept.append(Instance(None, masks[i], scores[i]))   # name None = not named yet
        small_kept.append(small)
        if len(kept) >= MAX_SEGMENTS:
            break
    print(f"  kept {len(kept)} segments of {len(masks)}")
    return kept


def crop_segment(image_arr, mask):
    """Crop the segment's bounding box, grey out everything else, pad to a square."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad_y, pad_x = int((y1 - y0) * 0.1), int((x1 - x0) * 0.1)
    y0, y1 = max(0, y0 - pad_y), min(mask.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(mask.shape[1], x1 + pad_x)
    patch = np.where(mask[..., None], image_arr, 128).astype(np.uint8)[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = patch
    return Image.fromarray(canvas)


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------
def clean_name(text):
    """Turn a model answer into a short object name, or '' if it isn't usable."""
    lines = text.strip().splitlines()
    t = lines[0] if lines else ""
    t = t.lower().strip(" .,:;!?\"'`*_-")
    for _ in range(3):   # strip stacked lead-ins such as "this is a ..."
        t = LEADING.sub("", t)
    t = t.strip(" .,:;!?\"'`*_-")
    if len(t) < 2 or t in BAD_NAMES or "unclear" in t:
        return ""
    if len(t.split()) > 4:   # a sentence, not a name
        return ""
    return t


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def load_vlm(model_id, device):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
        except TypeError:   # older transformers call this argument torch_dtype
            model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id)
    return model.to(device).eval()


class VlmNamer:
    """Open-ended naming: a small vision-language model says what each segment is."""

    def __init__(self, model_id, device):
        print(f"Loading the naming model {model_id}...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = load_vlm(model_id, device)

    @torch.no_grad()
    def describe_scene(self, image):
        """Create one concise, whole-image description to contextualize all segment names."""
        scene_image = image.copy()
        scene_image.thumbnail((VLM_CROP_SIZE, VLM_CROP_SIZE))
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": SCENE_PROMPT},
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=[scene_image], return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        description = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        description = " ".join(description.split())
        print(f"Whole-image context: {description or '(no description returned)'}")
        return description

    @torch.no_grad()
    def name_one(self, crop, scene_context=""):
        image = crop.resize((VLM_CROP_SIZE, VLM_CROP_SIZE))
        prompt_text = VLM_PROMPT.format(
            scene_context=scene_context or "No scene description is available."
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return clean_name(text)

    def name_crops(self, crops, scene_context=""):
        names = []
        for k, crop in enumerate(crops, 1):
            names.append(self.name_one(crop, scene_context))
            print(f"  {k}/{len(crops)}: {names[-1] or '(not a clear object, skipped)'}")
        return names


class ClipNamer:
    """Fast naming: CLIP picks the best match from a vocabulary (the word groups)."""

    def __init__(self, clip_name, vocab, device, min_conf):
        print(f"Loading CLIP ({clip_name}) with {len(vocab)} names...")
        self.device, self.vocab, self.min_conf = device, vocab, min_conf
        self.model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.base_prompts = vocab + REJECT_LABELS

    @torch.no_grad()
    def name_crops(self, crops, scene_context=""):
        # Include the scene in every text candidate so CLIP can disambiguate
        # visually similar crops (for example, a bowl in a kitchen vs. a hat).
        if scene_context:
            self.prompts = [
                f"a photo of {label} in a scene described as: {scene_context}"
                for label in self.base_prompts
            ]
        else:
            self.prompts = [f"a photo of a {label}" for label in self.base_prompts]
        names = []
        batch_size = 4 if self.device == "cpu" else 8
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            inputs = self.processor(text=self.prompts, images=batch, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits_per_image.detach().float().cpu().numpy()   # crops x prompts
            for row in softmax(logits):
                k = int(row.argmax())
                names.append(self.vocab[k] if k < len(self.vocab) and row[k] >= self.min_conf else "")
        print("  named: " + ", ".join(n or "(skipped)" for n in names))
        return names


# --------------------------------------------------------------------------
# Caching: segments once per photo, names per naming setup
# --------------------------------------------------------------------------
def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def segment_cache_path(cache_dir, image_path, args):
    with open(image_path, "rb") as f:
        image_hash = hashlib.sha1(f.read()).hexdigest()
    key = sha({"v": CACHE_VERSION, "image": image_hash, "sam": args.sam_model, "grid": args.grid,
               "max_side": args.max_side, "area": [MIN_AREA, MAX_AREA], "nms": NMS_IOU, "max": MAX_SEGMENTS})
    return Path(cache_dir) / f"segments_{key}.sam3cache.npz", key


def names_path(seg_path, namer_id):
    return seg_path.with_name(seg_path.name.replace(".sam3cache.npz", f".names_{namer_id}.json"))


def load_names(path, instances):
    try:
        names = json.loads(path.read_text(encoding="utf-8"))["names"]
        if len(names) != len(instances):
            return
        for inst, name in zip(instances, names):
            inst.name = name          # None = still unnamed, "" = rejected
    except Exception:
        pass


def save_names(path, instances):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"names": [i.name for i in instances]}), encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", help="PNG/JPEG path (omit for a file dialog)")
    p.add_argument("--sam-model", default=SAM_MODEL,
                   help="segmentation model (SAM 2.1 Small is the recommended CPU-friendly default)")
    p.add_argument("--grid", type=int, default=DEFAULT_GRID,
                   help="SAM prompts a GRID x GRID grid of points: increase to find smaller objects, at a speed cost")
    p.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                   help="maximum image dimension before processing; increase for more detail, decrease for speed")
    p.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 4),
                   help="PyTorch CPU threads (default: up to 8; try 6-8 on an Intel i7-1360P)")
    p.add_argument("--sam-batch-size", type=int, default=16,
                   help="SAM points per batch; lower reduces RAM use, higher may improve throughput")
    p.add_argument("--namer", choices=["vlm", "clip"], default="vlm",
                   help="vlm: open-ended names from a vision-language model; clip: pick from the word groups (faster)")
    p.add_argument("--vlm-model", default=VLM_MODEL, help="vision-language model used by --namer vlm")
    p.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP model used by --namer clip (Base is lighter than Large)")
    p.add_argument("--min-confidence", type=float, default=CLIP_MIN_CONF, help="--namer clip: minimum confidence")
    p.add_argument("--groups", default="auto",
                   help="--namer clip vocabulary: 'auto' (scene recognition), 'all', or a comma-separated list "
                        "of groups (general,closet,dresser,...)")
    p.add_argument("--auto-top", type=int, default=base.AUTO_TOP_GROUPS, help="with --groups auto: groups to add")
    p.add_argument("--scene-model", default=base.SCENE_MODEL, help="CLIP model used by --groups auto")
    p.add_argument("--no-cache", action="store_true", help="don't read or write cached results")
    p.add_argument("--recompute", action="store_true", help="ignore cached segments and names")
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "sam3_hidden_objects"))
    p.add_argument("--overlay", "--cheat", dest="overlay", action="store_true",
                   help="cheat mode: tint the hidden objects")
    return p


def prepare(path, args, stop_event=None):
    """Segment the photo (or load the segments from the cache) and set up the namer.

    Returns (image, instances, extras) where the instances are unnamed segments and
    extras holds the namer plus the already-named opening game, for show().
    """
    def check_cancel():
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

    # Configure CPU parallelism before model inference. Inter-op threads can only
    # be set once in some PyTorch builds, so tolerate an already-initialized runtime.
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(min(4, args.cpu_threads))
        except RuntimeError:
            pass

    image = Image.open(path).convert("RGB")
    image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
    shape = (image.height, image.width)
    device = pick_device()

    seg_path, seg_key = (None, None) if args.no_cache else segment_cache_path(args.cache_dir, path, args)
    instances = None
    if seg_path and seg_path.exists() and not args.recompute:
        instances = base.load_cache(seg_path, shape)
        if instances is not None:
            for inst in instances:
                inst.name = None
            print(f"Loaded {len(instances)} segments from cache, skipping segmentation")
    if instances is None:
        instances = segment(image, args.sam_model, args.grid, device, args.sam_batch_size)
        if seg_path and instances:
            base.save_cache(seg_path, instances, shape)
            print(f"Saved segments to cache: {seg_path}")
    check_cancel()
    if not instances:
        return image, [], {}

    # Tag the complete image once. The same description is supplied to every
    # segment-naming call, regardless of whether VLM or CLIP does the naming.
    scene_tagger = VlmNamer(args.vlm_model, device)
    scene_context = scene_tagger.describe_scene(image)
    check_cancel()

    # CLIP does not need the VLM weights after the scene description is made.
    # Release them before loading CLIP to reduce peak GPU memory use.
    if args.namer == "clip":
        del scene_tagger
        if device == "cuda":
            torch.cuda.empty_cache()

    # Naming back end
    if args.namer == "clip":
        groups = args.groups
        if groups == "auto":
            picked = detect_groups(image, device, args.auto_top, args.scene_model)
            groups = ",".join(["general"] + picked)
            print(f"Vocabulary groups: {groups}")
        vocab = load_concepts(None, groups)
        backend = ClipNamer(args.clip_model, vocab, device, args.min_confidence)
        namer_id = sha({
            "namer": "clip", "model": args.clip_model, "vocab": vocab,
            "conf": args.min_confidence, "scene_context": scene_context,
        })
    else:
        backend = scene_tagger
        namer_id = sha({
            "namer": "vlm", "model": args.vlm_model, "prompt": VLM_PROMPT,
            "scene_context": scene_context,
        })
    check_cancel()

    names_file = names_path(seg_path, namer_id) if seg_path else None
    if names_file and not args.recompute:
        load_names(names_file, instances)

    image_arr = np.array(image)

    def namer(batch):
        todo = [i for i in batch if i.name is None]
        if not todo:
            return
        print(f"Naming {len(todo)} segments...")
        crops = [crop_segment(image_arr, i.mask) for i in todo]
        for inst, name in zip(todo, backend.name_crops(crops, scene_context)):
            inst.name = name
        if names_file:
            save_names(names_file, instances)

    print("Choosing and naming the objects for the first game...")
    first_game = base.pick_chosen(instances, namer)
    return image, instances, {"namer": namer, "first_game": first_game}


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
