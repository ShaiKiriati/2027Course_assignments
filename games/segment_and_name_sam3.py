#!/usr/bin/env python3
"""
Find and name objects with SAM 3, pick 5-20 at random, and show the image
with the list of names below it. The display is interactive: left-click an
object in the image and its name turns red in the list while its outline is
drawn over the image.

SAM 3 segments by *concept*: you give it a short noun phrase ("dog", "coffee
mug") and it returns a mask for every matching instance, already named. So
there is no separate naming model; the program just runs SAM 3 over a list of
candidate concepts and keeps whatever it finds.

Setup:
    pip install -U transformers torch pillow numpy matplotlib accelerate
    # facebook/sam3 is a gated model: request access at
    # https://huggingface.co/facebook/sam3 , then log in:
    huggingface-cli login

Run:
    python segment_and_name_sam3.py                    # file dialog
    python segment_and_name_sam3.py photo.jpg
    python segment_and_name_sam3.py photo.jpg --concepts my_concepts.txt --overlay
    python segment_and_name_sam3.py photo.jpg --groups refrigerator --half   # fastest settings
"""
import argparse
import random
import time
import math
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL = "facebook/sam3"
MAX_SIDE = 1280        # downscale very large images for speed/memory
MIN_AREA = 0.001       # ignore instances smaller than this fraction of the image
DEDUPE_IOU = 0.7       # drop overlapping instances found under different concepts

# Candidate concepts. SAM 3 only looks for what you ask for, so edit freely
# or pass --concepts with one phrase per line. More concepts = slower.
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


def load_model(device, half):
    """Load SAM 3, optionally in half precision (CUDA only) for roughly 2x speed."""
    if half and device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using {dtype} precision")
        try:
            model = Sam3Model.from_pretrained(SAM3_MODEL, dtype=dtype)
        except TypeError:   # older transformers call this argument torch_dtype
            model = Sam3Model.from_pretrained(SAM3_MODEL, torch_dtype=dtype)
    else:
        if half:
            print("--half only applies on CUDA GPUs; using full precision")
        model = Sam3Model.from_pretrained(SAM3_MODEL)
    return model.to(device).eval()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def find_instances(image, concepts, device, threshold, half):
    """Run SAM 3 once per concept and collect every named instance.

    The expensive vision backbone runs ONCE for the image; each concept then only
    needs the (much lighter) text + detection + mask heads, reusing those embeddings.
    """
    print(f"Loading {SAM3_MODEL} on {device}...")
    if device == "cpu":
        print("  Warning: running on CPU. SAM 3 is very slow here; a CUDA GPU is strongly recommended.")
    model = load_model(device, half)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL)

    t0 = time.time()
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values.to(model.dtype))
    sync(device)
    print(f"Image encoded once in {time.time() - t0:.1f}s")
    sizes = img_inputs.get("original_sizes").tolist()

    found = []
    t1 = time.time()
    for i, name in enumerate(concepts, 1):
        text_inputs = processor(text=name, return_tensors="pt").to(device)
        outputs = model(vision_embeds=vision_embeds, **text_inputs)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5, target_sizes=sizes,
        )[0]
        for mask, score in zip(res["masks"], res["scores"]):
            m = mask.detach().cpu().numpy().astype(bool)
            if m.mean() >= MIN_AREA:
                found.append(Instance(name, m, float(score)))
        if i % 10 == 0 or i == len(concepts):
            per = (time.time() - t1) / i
            eta = per * (len(concepts) - i)
            print(f"  {i}/{len(concepts)} concepts, {len(found)} instances, "
                  f"{per:.2f}s/concept, ~{eta:.0f}s left", end="\r")
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
        ax_txt.text(0.5, 0.9, "No objects found. Try lowering --threshold or supplying more --concepts.",
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
    p.add_argument("--concepts", help="text file with one concept phrase per line (overrides --groups)")
    p.add_argument(
        "--groups", default="all",
        help="comma-separated concept groups: general,closet,dresser,refrigerator,dishwasher,pantry,suitcase "
             "(default: all)",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="SAM 3 detection threshold")
    p.add_argument("--half", action="store_true",
                   help="run in half precision on a CUDA GPU (faster, less memory)")
    p.add_argument("--batch", type=int, help=argparse.SUPPRESS)  # no longer used; kept so old commands still run
    p.add_argument("--overlay", action="store_true", help="tint the chosen objects on the image")
    args = p.parse_args()

    path = args.image or choose_file()
    if not path:
        print("No file selected.")
        return

    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    device = pick_device()

    concepts = load_concepts(args.concepts, args.groups)
    print(f"Searching for {len(concepts)} concepts")
    instances = find_instances(image, concepts, device, args.threshold, args.half)
    instances = dedupe(instances)
    print(f"{len(instances)} distinct objects found")

    n = min(random.randint(5, 20), len(instances))
    chosen = random.sample(instances, n) if n else []
    print("Chosen:", [i.name for i in chosen])
    show(image, chosen, args.overlay)


if __name__ == "__main__":
    main()
