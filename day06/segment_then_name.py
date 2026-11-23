#!/usr/bin/env python3
"""
Hidden Object Game, open-ended version: segment first, name afterwards.

1. SAM (SAM 3 by default) segments everything it can find in the photo, with no
   word list involved.
2. Each game picks 5-20 of those segments at random and names only those:
   - "vlm" (default): a vision-language model sees the full photo with the
     segment highlighted, plus a close-up, names it. CLIP quickly checks the
     proposed name; the VLM verifies only uncertain cases.
   - "clip": CLIP picks the best match from the word groups (much faster).
3. The names become the list in the game window. Click objects to find them.
   Press N for a new game: new random segments are picked and named.

Keep this file next to segment_and_name_sam3.py (it reuses the game window).

Setup (same environment as the other version):
    pip install -U transformers torch pillow numpy matplotlib accelerate

Run:
    python segment_then_name.py photo.jpg
    python segment_then_name.py photo.jpg --vlm-size 256M
    python segment_then_name.py photo.jpg --namer clip --groups bathroom,toys
    python segment_then_name.py photo.jpg --clip-diagnostic
    python segment_then_name.py photo.jpg --sam-model facebook/sam2.1-hiera-small   # no gated model, faster
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (AutoModelForImageTextToText, AutoProcessor, CLIPModel,
                          CLIPProcessor, pipeline)

import segment_and_name_sam3 as base
from segment_and_name_sam3 import (CONCEPT_GROUPS, Cancelled, Instance, choose_file,  # noqa: F401
                                   detect_groups, load_concepts, pick_device, show)

SAM_MODEL = "facebook/sam3"                       # segmentation; also works: facebook/sam2.1-hiera-small
VLM_MODELS = {
    "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
}
DEFAULT_VLM_SIZE = "500M"
CLIP_MODEL = "openai/clip-vit-large-patch14"       # naming with --namer clip
DEFAULT_GRID = 24       # SAM prompts a grid of GRID x GRID points; more = finds smaller things, slower
MAX_SIDE = base.MAX_SIDE
MAX_SEGMENTS = 80       # keep the best-scoring segments only
MIN_AREA, MAX_AREA = 0.004, 0.60   # allowed segment size (fraction of the image)
NMS_IOU = 0.6           # drop near-duplicate segments
CLIP_MIN_CONF = 0.15    # --namer clip: reject names CLIP is unsure about
CLIP_VERIFY_ACCEPT = 0.70
CLIP_VERIFY_REJECT = 0.25
CLIP_VERIFY_PROMPTS = (
    "a photo showing {name}",
    "a photo showing a different object, not {name}",
    "a photo showing only a small fragment of an object",
    "a photo of background, a texture, or a shadow",
    "a blurry or ambiguous photo where the object cannot be identified",
)
CROP_PADDING = 0.05     # fraction of the segment bounding box to include around the mask
VLM_CROP_SIZE = 512     # images shown to the VLM at this size
CACHE_VERSION = 1
REJECT_LABELS = ["background", "blurry background", "a small part of something", "a shadow", "a texture"]
VLM_PROMPT = ("You are shown two images of the same photo: first the full scene with the target segment "
              "highlighted in orange, then a close-up of that segment with everything outside its mask "
              "grayed out. Name only the object represented by the highlighted segment, using the full "
              "scene to understand its context. Give a short noun phrase of one to three words. If the "
              "segment is only a fragment, background, or ambiguous, do not guess; answer exactly: unclear")
VLM_VERIFY_PROMPT = ("The first image shows the full scene with a target segment highlighted in orange. "
                     "The second image is a close-up of that segment. The proposed name is {name!r}. "
                     "Does the highlighted segment clearly depict that object? Judge the segment itself, "
                     "using the scene only for context. Answer with exactly one word: yes, no, or unclear. "
                     "Use no if the name does not match the segment, and unclear if there is not enough "
                     "visual evidence.")
QUALITY_PROMPT_TEMPLATES = (
    "{word}",
    "a photo of {word}",
    "the {word} in the scene",
    "the visible {word}",
    "a segment showing {word}",
)

BAD_NAMES = {"unclear", "unknown", "none", "nothing", "background", "image", "picture", "photo",
             "texture", "pattern", "n/a", "na", "object", "item", "thing", "something",
             "it", "target", "target object", "target segment", "highlighted object", "highlighted segment"}
LEADING = re.compile(r"^(the|a|an|this is|it is|it's|there is|there's|image of|picture of|photo of|close up of)\s+")


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def segment(image, sam_model, grid, device):
    """Automatic mask generation (SAM prompts a grid of points), then filter and de-duplicate."""
    print(f"Segmenting with {sam_model} on {device} ({grid}x{grid} points)...")
    if device == "cpu":
        print("  On a CPU this step can take a few minutes. It only runs once per photo (results are cached).")
    generator = pipeline("mask-generation", model=sam_model, device=device)
    try:
        out = generator(image, points_per_batch=64, points_per_crop=grid)
    except (TypeError, ValueError):   # a transformers version without the grid option
        out = generator(image, points_per_batch=64)
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
    pad_y, pad_x = int((y1 - y0) * CROP_PADDING), int((x1 - x0) * CROP_PADDING)
    y0, y1 = max(0, y0 - pad_y), min(mask.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(mask.shape[1], x1 + pad_x)
    patch = np.where(mask[..., None], image_arr, 128).astype(np.uint8)[y0:y1, x0:x1]
    h, w = patch.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = patch
    return Image.fromarray(canvas)


def highlight_segment(image_arr, mask):
    """Tint a segment in the full photo so the VLM can use its scene context."""
    highlighted = image_arr.copy()
    tint = np.array([255, 128, 0], dtype=np.float32)
    highlighted[mask] = (highlighted[mask].astype(np.float32) * 0.4 + tint * 0.6).astype(np.uint8)
    return Image.fromarray(highlighted)


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------
def clean_name(text):
    """Turn a model answer into a short object name, or '' if it isn't usable."""
    lines = text.strip().splitlines()
    t = lines[0] if lines else ""
    t = t.lower().strip(" .,:;!?\"'`*_-")
    t = re.sub(r"^(?:object|label|name)\s*:\s*", "", t)
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
    """Open-ended naming with both full-scene context and a masked segment close-up."""

    def __init__(self, model_id, device, clip_verifier=None, clip_diagnostic=False):
        print(f"Loading the naming model {model_id}...")
        self.device = device
        self.clip_verifier = clip_verifier
        self.clip_diagnostic = clip_diagnostic
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = load_vlm(model_id, device)

    @torch.no_grad()
    def _ask(self, context_image, crop, question, max_new_tokens):
        context = ImageOps.pad(context_image, (VLM_CROP_SIZE, VLM_CROP_SIZE),
                               method=Image.Resampling.LANCZOS, color=(128, 128, 128))
        images = [context, crop.resize((VLM_CROP_SIZE, VLM_CROP_SIZE),
                                        resample=Image.Resampling.LANCZOS)]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

    def name_one(self, context_image, crop):
        return clean_name(self._ask(context_image, crop, VLM_PROMPT, max_new_tokens=8))

    def verify_name(self, context_image, crop, name):
        answer = self._ask(context_image, crop, VLM_VERIFY_PROMPT.format(name=name), max_new_tokens=4)
        token = answer.strip().lower().strip(" .,:;!?\"'`*_").split(maxsplit=1)
        return bool(token and token[0] == "yes"), answer

    def name_crops(self, crops, context_images, stop_event=None):
        names = []
        for k, (crop, context_image) in enumerate(zip(crops, context_images), 1):
            if stop_event is not None and stop_event.is_set():
                raise Cancelled()
            name = self.name_one(context_image, crop)
            if name:
                if self.clip_diagnostic:
                    rank, total, alternatives = self.clip_verifier.diagnose_name(crop, name)
                    top = ", ".join(f"{label} ({score:.1f})" for label, score in alternatives)
                    print(f"  CLIP diagnostic: {name!r} ranked {rank}/{total}; alternatives: {top}")
                verdict = "unclear"
                if self.clip_verifier is not None:
                    verdict, confidence = self.clip_verifier.assess_name(crop, name)
                    print(f"  CLIP check: {name!r} {verdict} ({confidence:.2f})")
                if verdict == "unclear":
                    supported, answer = self.verify_name(context_image, crop, name)
                    verdict = "yes" if supported else "no"
                    if verdict != "yes":
                        print(f"  {k}/{len(crops)}: {name!r} rejected by VLM verifier ({answer!r})")
                if verdict == "no":
                    name = ""
            names.append(name)
            print(f"  {k}/{len(crops)}: {name or '(not a clear object, skipped)'}")
        return names


class ClipNamer:
    """Fast naming: CLIP picks the best match from a vocabulary (the word groups)."""

    def __init__(self, clip_name, vocab, device, min_conf):
        if vocab:
            print(f"Loading CLIP ({clip_name}) with {len(vocab)} names on {device}...")
        else:
            print(f"Loading CLIP verifier ({clip_name}) on {device}...")
        self.device, self.vocab, self.min_conf = device, vocab, min_conf
        self.model = CLIPModel.from_pretrained(clip_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.prompts = [f"a photo of a {w}" for w in vocab + REJECT_LABELS]

    @torch.no_grad()
    def assess_name(self, crop, name):
        prompts = [template.format(name=name) for template in CLIP_VERIFY_PROMPTS]
        inputs = self.processor(text=prompts, images=crop, return_tensors="pt", padding=True).to(self.device)
        scores = self.model(**inputs).logits_per_image[0].detach().float().cpu().numpy()
        probabilities = softmax(scores)
        candidate_confidence = float(probabilities[0])
        if candidate_confidence >= CLIP_VERIFY_ACCEPT:
            verdict = "yes"
        elif candidate_confidence <= CLIP_VERIFY_REJECT:
            verdict = "no"
        else:
            verdict = "unclear"
        return verdict, candidate_confidence

    @torch.no_grad()
    def diagnose_name(self, crop, name):
        labels = [name] + [label for label in self.vocab + REJECT_LABELS
                           if label.casefold() != name.casefold()]
        prompts = [f"a photo of a {label}" for label in labels]
        inputs = self.processor(text=prompts, images=crop, return_tensors="pt", padding=True).to(self.device)
        scores = self.model(**inputs).logits_per_image[0].detach().float().cpu().numpy()
        rank = int(np.count_nonzero(scores[1:] >= scores[0])) + 1
        best = np.argsort(scores[1:])[-3:][::-1]
        alternatives = [(labels[i + 1], float(scores[i + 1])) for i in best]
        return rank, len(labels), alternatives

    @torch.no_grad()
    def name_crops(self, crops, stop_event=None):
        names = []
        total = len(crops)
        for start in range(0, len(crops), 8):
            if stop_event is not None and stop_event.is_set():
                raise Cancelled()
            batch = crops[start:start + 8]
            inputs = self.processor(text=self.prompts, images=batch, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits_per_image.detach().float().cpu().numpy()   # crops x prompts
            for row in softmax(logits):
                k = int(row.argmax())
                names.append(self.vocab[k] if k < len(self.vocab) and row[k] >= self.min_conf else "")
                print(f"  {len(names)}/{total}: {names[-1] or '(skipped)'}")
        return names


def _match_quality_masks(target_indices, targets, detected_masks, detected_scores):
    pairs = []
    for target_index in target_indices:
        for detection_index, detection_mask in enumerate(detected_masks):
            overlap = mask_iou(targets[target_index].mask, detection_mask)
            if overlap > 0:
                pairs.append((overlap, target_index, detection_index))
    pairs.sort(key=lambda pair: pair[0], reverse=True)

    matches, used_targets, used_detections = {}, set(), set()
    for overlap, target_index, detection_index in pairs:
        if target_index in used_targets or detection_index in used_detections:
            continue
        matches[target_index] = (overlap, detected_scores[detection_index])
        used_targets.add(target_index)
        used_detections.add(detection_index)
    return matches


@torch.no_grad()
def evaluate_naming_quality(image, targets, runs, device, stop_event=None):
    """Compare chosen named masks with SAM 3 detections under prompt variations."""
    if not targets:
        raise ValueError("There are no named segments in the current game to evaluate.")
    if not 1 <= runs <= len(QUALITY_PROMPT_TEMPLATES):
        raise ValueError(f"Run count must be between 1 and {len(QUALITY_PROMPT_TEMPLATES)}.")

    word_indices = {}
    for index, inst in enumerate(targets):
        if inst.name:
            word_indices.setdefault(inst.name, []).append(index)
    if not word_indices:
        raise ValueError("The current game has no usable names to evaluate.")

    results = [
        {"segment": index + 1, "word": inst.name, "mask_area": int(inst.mask.sum()), "runs": []}
        for index, inst in enumerate(targets)
    ]

    print(f"Starting SAM3 quality check: {len(targets)} segments, {runs} prompt runs")
    model = base.load_model(device, half=False, fast=False)
    processor = base.Sam3Processor.from_pretrained(base.SAM3_MODEL)
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    vision_embeds = model.get_vision_features(pixel_values=image_inputs.pixel_values.to(model.dtype))
    base.sync(device)
    target_sizes = image_inputs.get("original_sizes").tolist()
    word_count = len(word_indices)

    for run_index, template in enumerate(QUALITY_PROMPT_TEMPLATES[:runs], 1):
        for word_index, (word, indices) in enumerate(word_indices.items(), 1):
            if stop_event is not None and stop_event.is_set():
                raise Cancelled()
            prompt = template.format(word=word)
            text_inputs = processor(text=prompt, return_tensors="pt").to(device)
            outputs = model(vision_embeds=vision_embeds, **text_inputs)
            res = processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes,
            )[0]
            detected_masks = [mask.detach().cpu().numpy().astype(bool) for mask in res["masks"]]
            detected_scores = [float(score) for score in res["scores"]]
            limit = max(base.DEFAULT_MAX_PER_CONCEPT, len(indices))
            order = sorted(range(len(detected_scores)), key=detected_scores.__getitem__, reverse=True)[:limit]
            detected_masks = [detected_masks[i] for i in order]
            detected_scores = [detected_scores[i] for i in order]
            matches = _match_quality_masks(indices, targets, detected_masks, detected_scores)

            for index in indices:
                iou, confidence = matches.get(index, (0.0, 0.0))
                score = 100 * iou * confidence
                results[index]["runs"].append({
                    "run": run_index,
                    "prompt": prompt,
                    "iou": round(iou, 4),
                    "sam3_confidence": round(confidence, 4),
                    "score": round(score, 2),
                })
            print(f"Quality run {run_index}/{runs}: word {word_index}/{word_count} complete - {word}")

    for result in results:
        run_results = result["runs"]
        result["mean_iou"] = round(sum(item["iou"] for item in run_results) / runs, 4)
        result["mean_sam3_confidence"] = round(
            sum(item["sam3_confidence"] for item in run_results) / runs, 4)
        result["mean_score"] = round(sum(item["score"] for item in run_results) / runs, 2)

    del model, processor, image_inputs, vision_embeds
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "sam3_model": base.SAM3_MODEL,
        "runs": runs,
        "prompt_templates": list(QUALITY_PROMPT_TEMPLATES[:runs]),
        "score_definition": "Mean of 100 * mask IoU * SAM3 confidence across prompt runs.",
        "items": results,
    }


def save_quality_report(image_path, image_name, cache_dir, report):
    image_hash = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()[:16]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = Path(cache_dir) / "quality_reports" / f"naming_quality_{image_hash}_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    saved_report = dict(report)
    saved_report["image_name"] = image_name or Path(image_path).name
    saved_report["image_sha1"] = image_hash
    saved_report["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report_path.write_text(json.dumps(saved_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------
# Caching: segments once per photo, names per naming setup
# --------------------------------------------------------------------------
def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def segment_cache_path(cache_dir, image_path, args):
    with open(image_path, "rb") as f:
        image_hash = hashlib.sha1(f.read()).hexdigest()
    key = sha({"v": CACHE_VERSION, "image": image_hash, "sam": args.sam_model, "grid": args.grid,
               "max_side": MAX_SIDE, "area": [MIN_AREA, MAX_AREA], "nms": NMS_IOU, "max": MAX_SEGMENTS})
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
                   help="segmentation model (facebook/sam3, or e.g. facebook/sam2.1-hiera-small for speed)")
    p.add_argument("--grid", type=int, default=DEFAULT_GRID,
                   help="SAM prompts a GRID x GRID grid of points: more finds smaller objects but is slower")
    p.add_argument("--namer", choices=["vlm", "clip"], default="vlm",
                   help="vlm: open-ended names from a vision-language model; clip: pick from the word groups (faster)")
    p.add_argument("--vlm-size", choices=tuple(VLM_MODELS), default=DEFAULT_VLM_SIZE,
                   help="VLM size: 500M (default) or 256M (smaller and faster)")
    p.add_argument("--vlm-model", default=None,
                   help="custom vision-language model ID; overrides --vlm-size")
    p.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP model used by --namer clip")
    p.add_argument("--clip-diagnostic", action="store_true",
                   help="--namer vlm: print CLIP's best competing labels (diagnostic only; runs CLIP on CPU)")
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

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phone photos: apply the rotation tag
    image.thumbnail((MAX_SIDE, MAX_SIDE))
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
        instances = segment(image, args.sam_model, args.grid, device)
        if seg_path and instances:
            base.save_cache(seg_path, instances, shape)
            print(f"Saved segments to cache: {seg_path}")
    check_cancel()
    if not instances:
        return image, [], {}

    # Naming back end
    if args.namer == "clip":
        groups = args.groups
        if groups == "auto":
            picked = detect_groups(image, device, args.auto_top, args.scene_model)
            groups = ",".join(["general"] + picked)
            print(f"Vocabulary groups: {groups}")
        vocab = load_concepts(None, groups)
        backend = ClipNamer(args.clip_model, vocab, device, args.min_confidence)
        namer_id = sha({"namer": "clip", "model": args.clip_model, "vocab": vocab, "conf": args.min_confidence})
    else:
        vlm_model = args.vlm_model or VLM_MODELS[args.vlm_size]
        backend = None
        namer_id = sha({"namer": "vlm", "model": vlm_model, "prompt": VLM_PROMPT,
                        "verify_prompt": VLM_VERIFY_PROMPT, "image_size": VLM_CROP_SIZE,
                        "crop_padding": CROP_PADDING, "context_fit": "pad",
                        "name_cleanup": 2,
                        "clip_verify_prompts": CLIP_VERIFY_PROMPTS,
                        "clip_verify_thresholds": [CLIP_VERIFY_REJECT, CLIP_VERIFY_ACCEPT],
                        "clip_diagnostic": args.clip_diagnostic,
                        "clip_model": args.clip_model})
    check_cancel()

    names_file = names_path(seg_path, namer_id) if seg_path else None
    if names_file and not args.recompute:
        load_names(names_file, instances)

    image_arr = np.array(image)

    def namer(batch, stop=None):
        nonlocal backend
        todo = [i for i in batch if i.name is None]
        if not todo:
            return
        if stop is not None and stop.is_set():
            raise Cancelled()
        print(f"Naming {len(todo)} segments...")
        if backend is None:
            clip_vocab = load_concepts(None, "all") if args.clip_diagnostic else []
            clip_verifier = ClipNamer(args.clip_model, clip_vocab, device, args.min_confidence)
            backend = VlmNamer(vlm_model, device, clip_verifier, args.clip_diagnostic)
        crops = [crop_segment(image_arr, i.mask) for i in todo]
        if args.namer == "vlm":
            context_images = [highlight_segment(image_arr, i.mask) for i in todo]
            names = backend.name_crops(crops, context_images, stop)
        else:
            names = backend.name_crops(crops, stop)
        for inst, name in zip(todo, names):
            inst.name = name
        if names_file:
            save_names(names_file, instances)

    print("Choosing and naming the objects for the first game...")
    first_game = base.pick_chosen(instances, namer, stop_event)
    return image, instances, {"namer": namer, "first_game": first_game}


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.clip_diagnostic and args.namer != "vlm":
        parser.error("--clip-diagnostic can only be used with --namer vlm")
    path = args.image or choose_file()
    if not path:
        print("No file selected.")
        return
    image, instances, extras = prepare(path, args)
    show(image, instances, args.overlay, **extras)


if __name__ == "__main__":
    main()
