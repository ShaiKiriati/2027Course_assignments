#!/usr/bin/env python3
"""
SAM 2 version (no gated model needed).

1. SAM 2 automatically segments everything it can find in the image.
2. CLIP names each segment by picking the best match from a vocabulary.
3. 5-20 named objects are chosen at random.
4. The image is shown with the list of names below it. Left-click an object in
   the image: its name turns red in the list and its outline is drawn over the
   image.

Setup:
    pip install -U transformers torch pillow numpy matplotlib accelerate

Run:
    python segment_and_name_sam2.py                    # file dialog
    python segment_and_name_sam2.py photo.jpg --groups refrigerator --overlay
    python segment_and_name_sam2.py photo.jpg --sam-model facebook/sam2.1-hiera-small
"""
import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

SAM2_MODEL = "facebook/sam2.1-hiera-large"   # also: -base-plus, -small, -tiny (faster, less accurate)
CLIP_MODEL = "openai/clip-vit-large-patch14"  # or openai/clip-vit-base-patch32 (faster)
MAX_SIDE = 1024         # downscale big images for speed/memory
MAX_SEGMENTS = 60       # cap how many segments get named
MIN_AREA, MAX_AREA = 0.004, 0.70   # allowed segment size (fraction of image)
NMS_IOU = 0.6           # drop near-duplicate masks
MIN_CONFIDENCE = 0.15   # drop segments CLIP is unsure about (lots of labels, so probabilities are modest)
# Labels that "catch" junk segments (backgrounds, fragments). If one of these wins, the segment is dropped.
REJECT_LABELS = ["background", "blurry background", "a small part of something", "a shadow", "a texture"]

# Vocabulary CLIP picks names from. An object that is not in the list gets the
# closest wrong label, so edit freely or pass --concepts (one phrase per line).
DEFAULT_CONCEPTS = """
person, man, woman, child, dog, cat, horse, cow, sheep, bird, fish, elephant,
giraffe, zebra, bear, car, truck, bus, motorcycle, bicycle, boat, airplane,
train, traffic light, stop sign, bench, chair, sofa, table, desk, bed, pillow,
lamp, bookshelf, cabinet, door, window, curtain, mirror, rug, clock, vase,
potted plant, flower, tree, bush, laptop, monitor, keyboard, computer mouse,
phone, television, remote control, speaker, headphones, camera, cup, mug,
glass, bottle, bowl, plate, fork, knife, spoon, pot, pan, kettle, toaster,
microwave, oven, refrigerator, sink, toilet, towel, book, paper, pen, scissors,
backpack, handbag, suitcase, umbrella, hat, helmet, shirt, jacket, dress, pants,
shoe, glasses, watch, ball, kite, skateboard, surfboard, guitar, teddy bear,
apple, banana, orange, tomato, carrot, broccoli, bread, cake, pizza, sandwich,
fence, building, house, bridge, road, sign, flag, balloon, painting, basket,
box, bucket, candle, mountain, rock, cloud
"""

CLOSET = """
hanger, coat hanger, sweater, hoodie, t-shirt, jeans, skirt, blazer, suit jacket,
coat, raincoat, bathrobe, scarf, belt, tie, gloves, socks, cap, boots, sandals,
sneakers, high heels, slippers, shoe box, purse, duffel bag, folded clothes,
laundry basket, iron, ironing board, vacuum cleaner, storage bin
"""

DRESSER = """
folded shirt, folded pants, underwear, bra, pajamas, jewelry box, necklace,
bracelet, ring, earrings, perfume bottle, lipstick, makeup brush, hairbrush,
comb, hair dryer, tissue box, alarm clock, picture frame, trinket tray, drawer,
drawer handle, wallet, sunglasses, lotion bottle, jewelry
"""

REFRIGERATOR = """
milk carton, egg carton, eggs, butter, cheese, yogurt, jar, ketchup bottle,
mustard bottle, jam jar, juice bottle, soda can, beer bottle, water bottle,
food container, leftovers container, plastic wrap, lettuce, cucumber,
bell pepper, lemon, grapes, strawberries, blueberries, avocado, onion, garlic,
mushroom, ham, sausage, bacon, raw chicken, meat, tofu, tortillas, salad,
pickle jar, sauce bottle, mayonnaise jar, ice tray, freezer drawer, pie
"""

DISHWASHER = """
dishwasher rack, utensil basket, wine glass, drinking glass, coffee cup,
tea cup, saucer, dinner plate, serving bowl, mixing bowl, baking tray,
cutting board, spatula, ladle, whisk, tongs, colander, grater, peeler,
can opener, measuring cup, pot lid, food storage container, baby bottle,
chopsticks, sponge, dish soap, casserole dish, frying pan, teaspoon,
butter knife, steak knife, teapot, pitcher
"""

PANTRY = """
canned food, tin can, soup can, cereal box, pasta, spaghetti, bag of rice,
bag of flour, sugar, salt shaker, pepper shaker, spice jar, olive oil bottle,
vinegar bottle, honey jar, peanut butter jar, coffee beans, tea box, cookies,
crackers, bag of chips, granola bar, oatmeal, noodles, cooking oil, lentils,
nuts, dried fruit, popcorn, baking soda, broth carton, pasta sauce jar,
juice box, glass jar, storage jar, paper bag, plastic bag, shopping bag
"""

SUITCASE = """
suitcase, passport, travel pillow, toiletry bag, toothbrush, toothpaste,
shampoo bottle, razor, phone charger, charging cable, power adapter,
packing cube, luggage tag, swimsuit, flip flops, sunscreen, guidebook, map,
eye mask, hair straightener, deodorant, boarding pass, ticket, keys,
tablet, e-reader, power bank, zipper
"""

CONCEPT_GROUPS = {
    "general": DEFAULT_CONCEPTS,
    "closet": CLOSET,
    "dresser": DRESSER,
    "refrigerator": REFRIGERATOR,
    "dishwasher": DISHWASHER,
    "pantry": PANTRY,
    "suitcase": SUITCASE,
}


@dataclass
class Instance:
    name: str
    mask: np.ndarray   # bool, H x W
    score: float       # CLIP confidence in the name


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
        names = list(CONCEPT_GROUPS) if groups == "all" else [g.strip() for g in groups.split(",")]
        unknown = [g for g in names if g not in CONCEPT_GROUPS]
        if unknown:
            raise SystemExit(f"Unknown group(s): {unknown}. Choose from: {list(CONCEPT_GROUPS)}")
        items = []
        for g in names:
            items += [c.strip() for c in CONCEPT_GROUPS[g].replace("\n", " ").split(",")]
    seen, out = set(), []
    for c in items:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def segment(image, sam_model, device):
    """SAM 2 automatic mask generation, then filter by size and remove near-duplicates."""
    print(f"Segmenting with {sam_model} on {device}...")
    generator = pipeline("mask-generation", model=sam_model, device=device)
    out = generator(image, points_per_batch=64)
    masks = [torch.as_tensor(m).cpu().numpy().astype(bool) for m in out["masks"]]
    scores = [float(s) for s in out["scores"]]

    total = masks[0].size if masks else 1
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept, kept_small = [], []
    for i in order:
        if not (MIN_AREA <= masks[i].sum() / total <= MAX_AREA):
            continue
        small = masks[i][::4, ::4]   # downsample for a fast overlap check
        if any(mask_iou(small, k) > NMS_IOU for k in kept_small):
            continue
        kept.append(masks[i])
        kept_small.append(small)
        if len(kept) >= MAX_SEGMENTS:
            break
    print(f"  kept {len(kept)} of {len(masks)} masks")
    return kept


def crop_segment(image_arr, mask):
    """Crop the mask's bounding box, grey out the background, pad to a square."""
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


@torch.no_grad()
def name_segments(image, masks, vocab, clip_model, device, min_conf):
    """Name each segment with CLIP zero-shot classification over the vocabulary."""
    print(f"Naming {len(masks)} segments with CLIP...")
    model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model)

    labels = vocab + REJECT_LABELS
    prompts = [f"a photo of a {w}" for w in labels]
    t_in = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    t_feat = model.get_text_features(**t_in)
    t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)

    arr = np.array(image)
    crops = [crop_segment(arr, m) for m in masks]
    instances = []
    for start in range(0, len(crops), 16):
        batch = crops[start:start + 16]
        i_in = processor(images=batch, return_tensors="pt").to(device)
        i_feat = model.get_image_features(**i_in)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        probs = (model.logit_scale.exp() * i_feat @ t_feat.T).softmax(dim=-1)
        conf, idx = probs.max(dim=-1)
        for m, c, k in zip(masks[start:start + 16], conf.tolist(), idx.tolist()):
            if k < len(vocab) and c >= min_conf:
                instances.append(Instance(vocab[k], m, c))
    print(f"  {len(instances)} segments got a confident name")
    return instances


HIGHLIGHT_RGB = (255, 42, 42)
HIGHLIGHT_HEX = "#ff2a2a"
ROWS_PER_COLUMN = 7


def make_outline(mask, thickness):
    """Boolean image that is True on the inner border of `mask`, `thickness` px wide."""
    eroded = mask.copy()
    for _ in range(thickness):
        p = np.pad(eroded, 1, constant_values=False)
        eroded = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                  & p[1:-1, :-2] & p[1:-1, 2:])
    return mask & ~eroded


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


def show(image, chosen, overlay):
    """Interactive display: left-click an object in the image to highlight it
    (red outline on the image, red bold name in the list)."""
    base = np.asarray(image).astype(np.float32)
    h, w = base.shape[:2]

    if overlay and chosen:  # optional faint tint of every chosen object
        cmap = plt.get_cmap("tab20")
        for i, inst in enumerate(chosen):
            if inst.mask.shape == (h, w):
                color = np.array(cmap(i % 20)[:3]) * 255
                base[inst.mask] = base[inst.mask] * 0.6 + color * 0.4

    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.3])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(base.astype(np.uint8))
    ax_img.axis("off")
    # Transparent layer on top of the image; we paint the outline into it on click.
    empty = np.zeros((h, w, 4), dtype=np.uint8)
    outline_layer = ax_img.imshow(empty, interpolation="nearest")

    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")

    if not chosen:
        ax_txt.text(0.5, 0.9, "No objects found. Try lowering --min-confidence or supplying more --concepts.",
                    ha="center", va="top", fontsize=12, transform=ax_txt.transAxes)
        fig.tight_layout()
        plt.show()
        return

    ax_txt.text(0.5, 1.0, f"{len(chosen)} objects chosen at random: click one in the image to highlight it",
                ha="center", va="top", fontsize=11, transform=ax_txt.transAxes)

    labels = unique_labels(chosen)
    rows = min(ROWS_PER_COLUMN, len(labels))
    ncols = math.ceil(len(labels) / rows)
    col_w = 0.3
    left = 0.5 - ncols * col_w / 2
    default_color = plt.rcParams["text.color"]
    texts = []
    for i, label in enumerate(labels):
        col, row = divmod(i, rows)
        texts.append(ax_txt.text(
            left + col * col_w, 0.82 - row * 0.115, f"\u2022 {label}",
            ha="left", va="top", fontsize=11, color=default_color,
            transform=ax_txt.transAxes,
        ))

    areas = [inst.mask.sum() for inst in chosen]
    thickness = max(2, round(max(h, w) / 400))
    outline_cache = {}
    state = {"selected": None}

    def set_selected(idx):
        prev = state["selected"]
        if prev == idx:
            return
        if prev is not None:
            texts[prev].set_color(default_color)
            texts[prev].set_fontweight("normal")
        if idx is None:
            outline_layer.set_data(empty)
        else:
            texts[idx].set_color(HIGHLIGHT_HEX)
            texts[idx].set_fontweight("bold")
            if idx not in outline_cache:
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[make_outline(chosen[idx].mask, thickness)] = (*HIGHLIGHT_RGB, 255)
                outline_cache[idx] = rgba
            outline_layer.set_data(outline_cache[idx])
        state["selected"] = idx
        fig.canvas.draw_idle()

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax_img or event.xdata is None:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # zoom/pan tool is active; don't treat the click as a selection
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if not (0 <= x < w and 0 <= y < h):
            return
        hits = [i for i, inst in enumerate(chosen) if inst.mask[y, x]]
        # If objects overlap at this pixel, pick the smallest (most specific) one.
        # Clicking a pixel that belongs to no listed object clears the highlight.
        set_selected(min(hits, key=lambda i: areas[i]) if hits else None)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", help="PNG/JPEG path (omit for a file dialog)")
    p.add_argument("--concepts", help="text file with one object name per line (overrides --groups)")
    p.add_argument(
        "--groups", default="all",
        help="comma-separated vocabulary groups: general,closet,dresser,refrigerator,dishwasher,pantry,suitcase "
             "(default: all)",
    )
    p.add_argument("--sam-model", default=SAM2_MODEL, help="SAM 2 checkpoint on Hugging Face")
    p.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP checkpoint on Hugging Face")
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="minimum CLIP confidence for a name")
    p.add_argument("--overlay", action="store_true", help="tint the chosen objects on the image")
    args = p.parse_args()

    path = args.image or choose_file()
    if not path:
        print("No file selected.")
        return

    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    device = pick_device()

    masks = segment(image, args.sam_model, device)
    instances = []
    if masks:
        vocab = load_concepts(args.concepts, args.groups)
        print(f"Vocabulary: {len(vocab)} names")
        instances = name_segments(image, masks, vocab, args.clip_model, device, args.min_confidence)

    n = min(random.randint(5, 20), len(instances))
    chosen = random.sample(instances, n) if n else []
    print("Chosen:", [i.name for i in chosen])
    show(image, chosen, args.overlay)


if __name__ == "__main__":
    main()
