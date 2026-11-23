#!/usr/bin/env python3
"""
Hidden Object Game as a local web app (Flask).

Run it, and your browser opens the game:
    python hidden_objects_web.py
(or double-click play_hidden_objects_web.bat on Windows)

Everything runs on your own computer: the page is served from http://127.0.0.1:5000
and your photos never leave it.

To play on a phone or tablet on the same Wi-Fi network:
    python hidden_objects_web.py --host 0.0.0.0
and open the address it prints (http://<your computer's address>:5000) on the phone.
Do this only on a network you trust: anyone on that network can then open the game.

Needs, besides the libraries from the other versions:
    pip install flask

Keep this file in the same folder as segment_and_name_sam3.py and segment_then_name.py.
"""
import os

# Must be set before the heavy libraries are imported.
os.environ.setdefault("MPLBACKEND", "Agg")                    # no desktop windows are needed
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows without Developer Mode

import argparse
import contextlib
import importlib
import io
import json
import logging
import re
import socket
import tempfile
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image
from werkzeug.utils import secure_filename

MODULE_NAMES = {"list": "segment_and_name_sam3", "free": "segment_then_name"}
SAM_CHOICES = {
    "SAM 3 (most accurate)": "facebook/sam3",
    "SAM 2.1 base+ (faster)": "facebook/sam2.1-hiera-base-plus",
    "SAM 2.1 small (fast)": "facebook/sam2.1-hiera-small",
    "SAM 2.1 tiny (fastest)": "facebook/sam2.1-hiera-tiny",
}
DARKEN_RGBA = (0, 0, 0, 175)   # overlay on found objects
CHEAT_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40)]
TRAINING_COLORS = CHEAT_COLORS + [
    (128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128),
    (128, 0, 128), (0, 90, 160), (120, 70, 20), (70, 110, 20),
]


# --------------------------------------------------------------------------
# Helpers without any web or model code
# --------------------------------------------------------------------------
class LogBuffer:
    """Collects text like a terminal does: a carriage return makes the next text
    overwrite the current line (that is how progress lines update in place)."""

    def __init__(self, max_lines=500):
        self.lines = [""]
        self.overwrite = False
        self.max_lines = max_lines

    def feed(self, text):
        for part in re.split(r"(\r\n|\r|\n)", text):
            if part in ("\n", "\r\n"):
                self.lines.append("")
                self.overwrite = False
            elif part == "\r":
                self.overwrite = True
            elif part:
                if self.overwrite:
                    self.lines[-1] = ""
                    self.overwrite = False
                self.lines[-1] += part
        if len(self.lines) > self.max_lines:
            del self.lines[:-self.max_lines]

    def text(self):
        return "\n".join(self.lines)


def build_args(modules, s):
    """Validate the settings from the web page; return (module, args) for module.prepare()."""
    mode = s.get("mode", "list")
    if mode not in ("list", "free"):
        raise ValueError("Unknown kind of game.")
    free = mode == "free"
    try:
        pool = int(s.get("pool", 30))
        auto_top = int(s.get("auto_top", 3))
        threshold = float(s.get("threshold", 0.5))
        detail = int(s.get("detail", 24))
    except (TypeError, ValueError):
        raise ValueError("Please enter numbers in the number boxes.")
    quality_check = s.get("quality_check", False)
    if not isinstance(quality_check, bool):
        raise ValueError("Choose whether to run the SAM3 naming quality check.")
    quality_runs = 3
    if quality_check:
        raw_quality_runs = s.get("quality_runs", 3)
        try:
            quality_runs = int(raw_quality_runs)
        except (TypeError, ValueError):
            raise ValueError("Quality check runs must be a whole number between 1 and 5.")
        if isinstance(raw_quality_runs, bool) or quality_runs != raw_quality_runs:
            raise ValueError("Quality check runs must be a whole number between 1 and 5.")
    if quality_check and not free:
        raise ValueError("The SAM3 naming quality check is only available in open-ended mode.")
    if quality_check and not 1 <= quality_runs <= 5:
        raise ValueError("Quality check runs must be between 1 and 5.")
    naming = s.get("naming", "vlm")
    if naming not in ("vlm", "clip"):
        raise ValueError("Unknown naming method.")
    group_mode = s.get("group_mode", "auto")
    if group_mode not in ("auto", "all", "custom"):
        raise ValueError("Unknown word group option.")
    if auto_top < 1 or (not free and (pool < 0 or not 0 < threshold < 1)):
        raise ValueError("Stop-after must be 0 or more, groups at least 1, and the threshold between 0 and 1.")
    if free and not 8 <= detail <= 64:
        raise ValueError("Detail (points per side) should be between 8 and 64.")

    uses_groups = (not free) or naming == "clip"
    if uses_groups and group_mode == "custom":
        valid = set(modules["list"].CONCEPT_GROUPS)
        picked = [g for g in s.get("groups", []) if g in valid]
        if not picked:
            raise ValueError("Tick at least one word group, or choose another option.")
        groups = ",".join(picked)
    else:
        groups = group_mode if uses_groups else "auto"

    module = modules[mode]
    args = module.build_parser().parse_args([])
    args.groups = groups
    args.auto_top = auto_top
    args.no_cache = not bool(s.get("use_cache", True))
    args.overlay = bool(s.get("cheat", False))
    args.quality_check = quality_check
    args.quality_runs = quality_runs
    args.clip_diagnostic = free and naming == "vlm" and bool(s.get("clip_diagnostic", False))
    if free:
        vlm_size = s.get("vlm_size", modules["free"].DEFAULT_VLM_SIZE)
        if vlm_size not in modules["free"].VLM_MODELS:
            raise ValueError("Choose the 500M or 256M vision-language model.")
        args.namer = naming
        args.vlm_size = vlm_size
        args.sam_model = SAM_CHOICES.get(s.get("sam"), next(iter(SAM_CHOICES.values())))
        args.grid = detail
    else:
        args.fast = bool(s.get("fast", True))
        args.half = bool(s.get("half", False))
        args.pool = pool
        args.threshold = threshold
    return module, args


def png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------
# The one running game (this is a single-user app on your own computer)
# --------------------------------------------------------------------------
class Busy(Exception):
    pass


class Stale(Exception):
    pass


class LogWriter:
    """File-like object that sends everything written to it into the session log."""

    def __init__(self, session):
        self.session = session

    def write(self, text):
        if text:
            with self.session.lock:
                self.session.log.feed(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.phase = "loading"        # loading -> idle -> working -> ready/report
        self.kind = ""                # what the background job is doing: "find" or "newgame"
        self.job_id = 0
        self.job_started_at = 0.0
        self.quality_report = None
        self.quality_report_path = None
        self.error = ""
        self.error_id = 0
        self.load_error = ""
        self.modules = None
        self.log = LogBuffer()
        self.stop_event = threading.Event()
        self.image = None
        self.instances = []
        self.extras = {}
        self.cheat = False
        self.chosen, self.labels, self.areas = [], [], []
        self.found = set()
        self.training_mode = False
        self.training_items = []
        self.training_saved = set()
        self.game_id = 0
        self.png_cache = {}

    # ---- start-up ------------------------------------------------------
    def load_modules(self):
        """Import torch and friends in the background so the page appears at once."""
        try:
            mods = {key: importlib.import_module(name) for key, name in MODULE_NAMES.items()}
            with self.lock:
                self.modules = mods
                self.phase = "idle"
        except BaseException:
            traceback.print_exc()
            with self.lock:
                self.load_error = traceback.format_exc().strip().splitlines()[-1]

    # ---- background jobs -----------------------------------------------
    def start_job(self, kind, func):
        with self.lock:
            if self.phase in ("loading", "working"):
                raise Busy("Please wait until the current job has finished.")
            self.phase = "working"
            self.kind = kind
            self.job_id += 1
            self.job_started_at = time.monotonic()
            self.quality_report = None
            self.quality_report_path = None
            self.log = LogBuffer()
            self.stop_event = threading.Event()
        threading.Thread(target=self._run, args=(func,), daemon=True).start()

    def _run(self, func):
        writer = LogWriter(self)
        error = ""
        cancelled = False
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                func()
        except BaseException as e:
            if isinstance(e, self.modules["list"].Cancelled):
                cancelled = True
                writer.write("\nCancelled.\n")
            else:
                text = traceback.format_exc()
                writer.write("\n" + text)
                error = text.strip().splitlines()[-1]
        with self.lock:
            if error:
                self.error = error
                self.error_id += 1
            if not error and not cancelled and self.kind == "quality" and self.quality_report is not None:
                self.phase = "report"
            else:
                # After a failed or cancelled job we fall back to the previous game, if there was one.
                self.phase = "ready" if self.image is not None else "idle"

    def find_objects(self, module, path, args):
        """Job: find the objects in a photo and open the first game."""
        def job():
            image, instances, extras = module.prepare(path, args, self.stop_event)
            first = extras.get("first_game") or self.modules["list"].pick_chosen(
                instances, extras.get("namer"), self.stop_event)
            with self.lock:
                self.image, self.instances, self.extras = image, instances, extras
                self.cheat = bool(args.overlay)
                self.begin_game(first)
        self.start_job("find", job)

    def find_training_items(self, module, path, args):
        """Prepare CLIP labels and retain only rejected/uncertain items for review."""
        def job():
            image, instances, extras = module.prepare(path, args, self.stop_event)
            items = extras.get("training_items", [])
            with self.lock:
                self.image, self.instances, self.extras = image, instances, extras
                self.cheat = False
                self.training_mode = True
                self.training_items = list(items)
                self.training_saved = set()
                self.chosen = [inst for inst, _status, _groups in items]
                self.labels = [inst.name for inst in self.chosen]
                self.areas = [int(inst.mask.sum()) for inst in self.chosen]
                self.found = set()
                self.png_cache = {}
                self.game_id += 1
        self.start_job("training", job)

    def check_naming_quality(self, module, path, args):
        def job():
            image, _, extras = module.prepare(path, args, self.stop_event)
            targets = extras.get("first_game", [])
            del extras
            report = module.evaluate_naming_quality(
                image, targets, args.quality_runs, self.modules["list"].pick_device(), self.stop_event)
            report_path = module.save_quality_report(
                path, args.quality_image_name, args.cache_dir, report)
            with self.lock:
                self.quality_report = report
                self.quality_report_path = report_path
            print(f"Saved quality report: {report_path}")
        self.start_job("quality", job)

    def new_game(self):
        """Job: pick (and, in open-ended mode, name) a fresh random set of objects."""
        def job():
            with self.lock:
                instances, namer = self.instances, self.extras.get("namer")
            chosen = self.modules["list"].pick_chosen(instances, namer, self.stop_event)
            with self.lock:
                self.begin_game(chosen)
        with self.lock:
            if self.image is None:
                raise Stale("There is no photo yet.")
        self.start_job("newgame", job)

    def cancel(self):
        self.stop_event.set()

    # ---- game state ------------------------------------------------------
    def begin_game(self, chosen):
        self.training_mode = False
        self.training_items = []
        self.training_saved = set()
        self.chosen = list(chosen)
        self.labels = self.modules["list"].unique_labels(self.chosen) if self.chosen else []
        self.areas = [int(i.mask.sum()) for i in self.chosen]
        self.found = set()
        self.png_cache = {}
        self.game_id += 1

    def click(self, x, y, game_id):
        with self.lock:
            if self.image is None or game_id != self.game_id:
                raise Stale("This game has changed. Reload the page.")
            h, w = self.image.height, self.image.width
            px = min(max(int(x * w), 0), w - 1)
            py = min(max(int(y * h), 0), h - 1)
            # Smallest object under the click wins; found objects and empty space do nothing.
            hits = [i for i, inst in enumerate(self.chosen) if inst.mask[py, px]]
            result = {"hit": None, "new": False}
            if hits:
                target = min(hits, key=lambda i: self.areas[i])
                result["hit"] = target
                if target not in self.found:
                    self.found.add(target)
                    result["new"] = True
            result.update(found=len(self.found), total=len(self.chosen),
                          done=len(self.found) == len(self.chosen) and len(self.chosen) > 0)
            return result

    def clue(self, game_id):
        with self.lock:
            if self.image is None or game_id != self.game_id:
                raise Stale("This game has changed. Reload the page.")
            remaining = [i for i in range(len(self.chosen)) if i not in self.found]
            if not remaining:
                return {"clue": None}

            idx = int(np.random.choice(remaining))
            mask = self.chosen[idx].mask
            ys, xs = np.nonzero(mask)
            point = int(np.random.randint(len(xs)))
            h, w = mask.shape
            side = min(max(1, int(min(w, h) * 0.35)), w, h)
            left = min(max(0, int(xs[point] - side / 2)), w - side)
            top = min(max(0, int(ys[point] - side / 2)), h - side)
            return {"clue": {"x": left / w, "y": top / h,
                             "width": side / w, "height": side / h}}

    def overlay_png(self, game_id, idx):
        with self.lock:
            if game_id != self.game_id or not 0 <= idx < len(self.chosen):
                return None
            key = ("dark", idx)
            if key not in self.png_cache:
                mask = self.chosen[idx].mask
                rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
                rgba[mask] = DARKEN_RGBA
                self.png_cache[key] = png_bytes(rgba)
            return self.png_cache[key]

    def cheat_png(self, game_id):
        with self.lock:
            if game_id != self.game_id or not self.chosen:
                return None
            if "cheat" not in self.png_cache:
                rgba = np.zeros(self.chosen[0].mask.shape + (4,), dtype=np.uint8)
                for i, inst in enumerate(self.chosen):
                    rgba[inst.mask] = (*CHEAT_COLORS[i % len(CHEAT_COLORS)], 110)
                self.png_cache["cheat"] = png_bytes(rgba)
            return self.png_cache["cheat"]

    def training_review(self):
        with self.lock:
            if not self.training_mode or self.image is None:
                return None
            return {
                "game_id": self.game_id,
                "width": self.image.width,
                "height": self.image.height,
                "items": [
                    {
                        "index": index,
                        "label": instance.name,
                        "status": status,
                        "source_groups": list(source_groups),
                        "color": "#{:02x}{:02x}{:02x}".format(*TRAINING_COLORS[index % len(TRAINING_COLORS)]),
                        "saved": index in self.training_saved,
                    }
                    for index, (instance, status, source_groups) in enumerate(self.training_items)
                ],
            }

    def training_outline_png(self, game_id):
        with self.lock:
            if not self.training_mode or game_id != self.game_id or self.image is None:
                return None
            key = "training-outline"
            if key not in self.png_cache:
                h, w = self.image.height, self.image.width
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                for index, (instance, _status, _source_groups) in enumerate(self.training_items):
                    mask = instance.mask
                    padded = np.pad(mask, 1, constant_values=False)
                    interior = (mask & padded[:-2, 1:-1] & padded[2:, 1:-1]
                                & padded[1:-1, :-2] & padded[1:-1, 2:])
                    outline = mask & ~interior
                    rgba[outline] = (*TRAINING_COLORS[index % len(TRAINING_COLORS)], 255)
                self.png_cache[key] = png_bytes(rgba)
            return self.png_cache[key]

    def save_training_word(self, game_id, index, word, group):
        with self.lock:
            if not self.training_mode or game_id != self.game_id:
                raise Stale("This training review has changed. Reload the page.")
            if not 0 <= index < len(self.training_items):
                raise Stale("This training label no longer exists.")
            if self.modules is None:
                raise Stale("The CLIP model module is not loaded.")
            instance, _status, source_groups = self.training_items[index]
            if not isinstance(group, str) or group not in source_groups:
                raise ValueError("Choose one of the source groups for this label.")
            normalized, added = self.modules["list"].save_concept_word(group, word)
            instance.name = normalized
            self.training_saved.add(index)
            return {"word": normalized, "group": group, "added": added, "index": index}

    def photo_jpeg(self):
        with self.lock:
            if self.image is None:
                return None
            buf = io.BytesIO()
            self.image.save(buf, "JPEG", quality=92)
            return buf.getvalue()

    # ---- what the page asks for ---------------------------------------------
    def info(self):
        with self.lock:
            if self.modules is None:
                return {"ready": False, "load_error": self.load_error}
            device = self.modules["list"].pick_device()
            if device == "cuda" and getattr(self.modules["list"].torch.version, "hip", None):
                label = "an AMD graphics card via ROCm (fast)"
            else:
                label = {"cuda": "an NVIDIA graphics card (fast)", "mps": "the Apple graphics chip"}.get(
                    device, "the CPU only (slow, so keep Fast mode on and expect some waiting)")
            return {"ready": True, "device_label": f"Ready. The models will run on {label}.",
                    "groups": list(self.modules["list"].CONCEPT_GROUPS), "sam_choices": list(SAM_CHOICES)}

    def status(self):
        with self.lock:
            return {"phase": self.phase, "kind": self.kind, "log": self.log.text(), "game_id": self.game_id,
                    "error": self.error, "error_id": self.error_id, "job_id": self.job_id,
                    "elapsed_seconds": time.monotonic() - self.job_started_at if self.phase == "working" else 0,
                    "quality_report": self.quality_report if self.phase == "report" else None,
                    "quality_report_path": str(self.quality_report_path) if self.phase == "report" else None}

    def game(self):
        with self.lock:
            if self.image is None:
                return None
            return {"game_id": self.game_id, "width": self.image.width, "height": self.image.height,
                    "items": list(self.labels), "found": sorted(self.found), "cheat": self.cheat}


SESSION = Session()
UPLOAD_DIR = None
upload_counter = 0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
app.config["TRAINING_MODE"] = False


@app.after_request
def no_store(resp):
    if request.path.startswith("/api/") and not request.path.endswith(".png"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    if app.config.get("TRAINING_MODE"):
        response = send_file(Path(__file__).with_name("hidden_objects_training.html"),
                             mimetype="text/html", max_age=0)
        response.headers["Cache-Control"] = "no-store"
        return response
    return Response(INDEX_HTML, mimetype="text/html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/info")
def api_info():
    info = SESSION.info()
    info["training_mode"] = bool(app.config.get("TRAINING_MODE"))
    phone_url = app.config.get("PHONE_URL")
    if phone_url:
        info["phone_url"] = phone_url
    return jsonify(info)


@app.get("/api/status")
def api_status():
    return jsonify(SESSION.status())


@app.get("/api/quality_report")
def api_quality_report():
    with SESSION.lock:
        report_path = SESSION.quality_report_path
    if report_path is None or not report_path.is_file():
        return jsonify(error="No saved quality report is available."), 404
    return send_file(report_path, as_attachment=True, download_name=report_path.name,
                     mimetype="application/json")


@app.post("/api/start")
def api_start():
    global upload_counter, UPLOAD_DIR
    photo = request.files.get("photo")
    if photo is None or not photo.filename:
        return jsonify(error="Please choose a photo (PNG or JPEG) first."), 400
    suffix = Path(secure_filename(photo.filename)).suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        return jsonify(error="Please choose a PNG or JPEG photo."), 400
    try:
        settings = json.loads(request.form.get("settings", "{}"))
        if not isinstance(settings, dict):
            return jsonify(error="Settings must be a JSON object."), 400
        if app.config.get("TRAINING_MODE"):
            settings.update(mode="free", naming="clip", quality_check=False)
        if SESSION.modules is None:
            return jsonify(error="The libraries are still loading. Please wait a moment."), 409
        module, args = build_args(SESSION.modules, settings)
    except ValueError as e:      # includes bad JSON
        return jsonify(error=str(e)), 400
    if UPLOAD_DIR is None:
        UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="hidden_objects_"))
    upload_counter += 1
    path = UPLOAD_DIR / f"upload_{upload_counter}{suffix}"
    photo.save(path)
    args.quality_image_name = secure_filename(photo.filename)
    try:
        if app.config.get("TRAINING_MODE"):
            args.training_mode = True
            SESSION.find_training_items(module, str(path), args)
        elif args.quality_check:
            SESSION.check_naming_quality(module, str(path), args)
        else:
            SESSION.find_objects(module, str(path), args)
    except Busy as e:
        return jsonify(error=str(e)), 409
    return jsonify(ok=True)


@app.get("/api/training")
def api_training():
    review = SESSION.training_review()
    if review is None:
        return jsonify(error="There is no training review yet."), 404
    return jsonify(review)


@app.get("/api/training/outline/<int:game_id>.png")
def api_training_outline(game_id):
    data = SESSION.training_outline_png(game_id)
    return Response(data, mimetype="image/png") if data else ("", 404)


@app.post("/api/training/learn")
def api_training_learn():
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify(error="Only the computer running the trainer can update its vocabulary."), 403
    body = request.get_json(silent=True) or {}
    try:
        game_id, index, word, group = (
            int(body["game_id"]), int(body["index"]), body["word"], body["group"]
        )
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Provide a training game, label index, corrected word, and group."), 400
    try:
        result = SESSION.save_training_word(game_id, index, word, group)
    except Stale as e:
        return jsonify(error=str(e)), 409
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except (OSError, RuntimeError) as e:
        return jsonify(error=f"Could not update the CLIP word groups: {e}"), 500
    return jsonify(result)


@app.post("/api/new_game")
def api_new_game():
    try:
        SESSION.new_game()
    except (Busy, Stale) as e:
        return jsonify(error=str(e)), 409
    return jsonify(ok=True)


@app.post("/api/cancel")
def api_cancel():
    SESSION.cancel()
    return jsonify(ok=True)


@app.get("/api/game")
def api_game():
    game = SESSION.game()
    if game is None:
        return jsonify(error="There is no game yet."), 404
    return jsonify(game)


@app.get("/api/image")
def api_image():
    data = SESSION.photo_jpeg()
    if data is None:
        return "", 404
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/click")
def api_click():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(SESSION.click(float(body["x"]), float(body["y"]), int(body["game_id"])))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Bad click."), 400
    except Stale as e:
        return jsonify(error=str(e)), 409


@app.post("/api/clue")
def api_clue():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(SESSION.clue(int(body["game_id"])))
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Bad clue request."), 400
    except Stale as e:
        return jsonify(error=str(e)), 409


@app.get("/api/overlay/<int:game_id>/<int:idx>.png")
def api_overlay(game_id, idx):
    data = SESSION.overlay_png(game_id, idx)
    return Response(data, mimetype="image/png") if data else ("", 404)


@app.get("/api/cheat/<int:game_id>.png")
def api_cheat(game_id):
    data = SESSION.cheat_png(game_id)
    return Response(data, mimetype="image/png") if data else ("", 404)


@app.post("/api/quit")
def api_quit():
    if request.remote_addr not in ("127.0.0.1", "::1"):   # a phone on the network must not stop the game
        return jsonify(error="Only the computer that runs the game can stop it."), 403
    threading.Timer(0.4, os._exit, args=(0,)).start()
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hidden Object Game</title>
<style>
  :root { --bg:#f4f6f9; --card:#ffffff; --text:#1c2230; --muted:#5f6b7c; --accent:#2f6fed; --border:#d5dae3; --found:#98a1b0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151b; --card:#1b2029; --text:#e8ebf2; --muted:#9aa3b2; --accent:#6b9bff; --border:#323a49; --found:#6c7585; }
  }
  [hidden] { display:none !important; }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.45 system-ui, "Segoe UI", Roboto, sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; align-items:center; justify-content:space-between; padding:10px 20px; background:var(--card); border-bottom:1px solid var(--border); }
  h1 { font-size:1.15rem; margin:0; }
  main { max-width:1040px; margin:0 auto; padding:16px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin-bottom:12px; }
  .card h2 { font-size:.9rem; margin:0 0 8px; color:var(--muted); font-weight:600; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .grid > .card { margin-bottom:0; }
  @media (max-width:780px) { .grid { grid-template-columns:1fr; } }
  label.row { display:block; margin:4px 0; }
  label.num { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:4px 0; }
  input[type=number] { width:5.5em; font:inherit; padding:3px 6px; }
  select { font:inherit; padding:3px 6px; max-width:100%; }
  .off { opacity:.45; }
  .note { color:var(--muted); font-size:.9rem; margin:6px 0 0; }
  .error { color:#c62828; margin:8px 0 0; }
  @media (prefers-color-scheme: dark) { .error { color:#ff8a80; } }
  button { font:inherit; padding:8px 16px; border-radius:8px; border:1px solid var(--border); background:var(--card); color:var(--text); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  button.link { border:none; background:none; color:var(--muted); padding:4px 8px; }
  #quit { background:#c62828; border-color:#c62828; color:#fff; font-weight:600; }
  #quit:hover { background:#a91f1f; border-color:#a91f1f; }
  #clue { background:#ffd400; border-color:#ffd400; color:#1c2230; font-weight:600; }
  #clue:hover { background:#e6bf00; border-color:#e6bf00; }
  #clue:disabled { opacity:1; }
  .actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  .spacer { flex:1; }
  #photoPreview { max-height:90px; max-width:160px; border-radius:6px; vertical-align:middle; margin-left:10px; }
  .progress-status { margin:8px 0; }
  progress { display:block; width:100%; height:16px; accent-color:var(--accent); }
  .quality-table-wrap { overflow-x:auto; }
  .quality-table { border-collapse:collapse; width:100%; }
  .quality-table th,.quality-table td { border-bottom:1px solid var(--border); padding:7px; text-align:left; vertical-align:top; }
  #stage { position:relative; display:inline-block; max-width:100%; cursor:pointer; line-height:0; user-select:none; -webkit-user-select:none; touch-action:manipulation; -webkit-tap-highlight-color:transparent; }
  #photo { display:block; max-width:100%; max-height:68vh; width:auto; height:auto; border-radius:6px; }
  #stage img.ov { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  #clueBox { position:absolute; border:4px solid #ffd400; border-radius:4px; background:rgba(255,212,0,.14); box-shadow:0 0 0 2px rgba(0,0,0,.55); pointer-events:none; z-index:2; }
  #imageViewport { max-width:100%; max-height:68vh; overflow:auto; }
  #stage { transform-origin:top left; }
  #zoomLevel { min-width:4em; text-align:center; }
  #items { columns:3 210px; list-style:none; padding:0; margin:14px 0 0; }
  #items li { padding:2px 0; break-inside:avoid; }
  #items li.found { text-decoration:line-through; color:var(--found); }
  #banner { background:var(--accent); color:#fff; border-radius:8px; padding:10px 14px; margin-bottom:10px; font-weight:600; }
  #toast { position:fixed; left:50%; top:14px; transform:translateX(-50%); max-width:90vw; background:#c62828; color:#fff; padding:10px 16px; border-radius:8px; z-index:10; box-shadow:0 4px 14px rgba(0,0,0,.3); }
</style>
</head>
<body>
<header>
  <h1>Hidden Object Game</h1>
  <button id="quit" title="Stop the program">Quit</button>
</header>
<div id="toast" hidden></div>
<main>

  <!-- Menu -->
  <section id="menu">
    <div class="card">
      <h2>Photo</h2>
      <input type="file" id="photoFile" accept="image/png,image/jpeg"><img id="photoPreview" alt="" hidden>
    </div>

    <div class="grid">
      <div>
        <div class="card" id="modeBox">
          <h2>What kind of game?</h2>
          <label class="row"><input type="radio" name="mode" value="list" checked> Search for objects from a word list (requires SAM3)</label>
          <label class="row"><input type="radio" name="mode" value="free"> Segment first, then name random segments</label>
        </div>
        <div class="card" id="freeBox" style="margin-top:12px">
          <h2>Open-ended mode settings</h2>
          <label class="row"><input type="radio" name="naming" value="vlm" checked> Vision-language model (open-ended names, slower)</label>
          <label class="row"><input type="radio" name="naming" value="clip"> CLIP with the word groups (faster)</label>
          <label class="num" id="vlmSizeBox">VLM size <select id="vlmSize">
            <option value="500M">SmolVLM-500M (default; better, slower)</option>
            <option value="256M">SmolVLM-256M (smaller, faster)</option>
          </select></label>
          <label class="row" id="clipDiagnosticBox"><input type="checkbox" id="clipDiagnostic"> Print CLIP alternatives for VLM names (CPU diagnostic only)</label>
          <div class="card" id="qualityBox" style="margin-top:10px">
            <label class="row"><input type="checkbox" id="qualityCheck"> Run SAM3 naming quality check instead of playing</label>
            <label class="num" id="qualityRunsBox">Prompt runs (1-5) <input type="number" id="qualityRuns" min="1" max="5" value="3"></label>
            <p class="note">Uses SAM3 (requires model access), compares each target mask with SAM3's match, and saves a JSON report. This can take a long time.</p>
          </div>
          <label class="num">Segmentation model <select id="sam"></select></label>
          <label class="num">Detail (points per side) <input type="number" id="detail" min="8" max="64" value="24"></label>
        </div>
      </div>
      <div>
        <div class="card" id="groupBox">
          <h2>Word groups</h2>
          <label class="row"><input type="radio" name="group_mode" value="auto" checked> Recognise the scene and pick groups (recommended)</label>
          <label class="num" style="padding-left:22px">Groups to add <input type="number" id="auto_top" min="1" max="6" value="3"></label>
          <label class="row"><input type="radio" name="group_mode" value="all"> Every group (slowest)</label>
          <label class="row"><input type="radio" name="group_mode" value="custom"> Only these groups:</label>
          <div id="groupBoxes" style="padding-left:22px"></div>
        </div>
        <div class="card" id="speedBox" style="margin-top:12px">
          <h2>Speed and quality (SAM3)</h2>
          <label class="row"><input type="checkbox" id="fast" checked> Fast mode (lower resolution; good on a CPU)</label>
          <label class="row"><input type="checkbox" id="half"> Half precision (NVIDIA graphics cards only)</label>
          <label class="num">Stop after this many objects (0 = all) <input type="number" id="pool" min="0" max="500" value="30"></label>
          <label class="num">Detection threshold <input type="number" id="threshold" min="0.1" max="0.9" step="0.05" value="0.5"></label>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:12px">
      <label class="row"><input type="checkbox" id="useCache" checked> Reuse saved results for this photo</label>
      <label class="row"><input type="checkbox" id="cheat"> Cheat mode (shows where the hidden objects are)</label>
    </div>

    <div class="actions">
      <button class="primary" id="start" disabled>Start game</button>
      <button id="resume" hidden>Back to the game</button>
      <span class="note" id="deviceNote">Loading application libraries&hellip;</span>
    </div>
    <div id="startupProgress" class="card" role="status" aria-live="polite">
      <p class="progress-status" id="startupText">Loading application libraries&hellip;</p>
      <progress id="startupBar" aria-label="Loading application libraries"></progress>
    </div>
    <p id="phoneAccess" class="note" hidden>
      To play on a phone on this Wi-Fi, open <a id="phoneUrl"></a>. Only use this link on a trusted network.
    </p>
    <p class="error" id="menuError" hidden></p>
  </section>

  <!-- Working -->
  <section id="working" hidden>
    <div class="card">
      <h2 id="workTitle">Working&hellip;</h2>
      <p class="progress-status" id="jobProgressText" role="status" aria-live="polite">Preparing&hellip;</p>
      <progress id="jobProgressBar" max="100" aria-label="Game preparation progress"></progress>
      <div class="actions"><button id="cancel">Cancel</button>
        <span class="note" id="workNote">The game opens automatically when it is ready.</span></div>
    </div>
  </section>

  <section id="qualityReport" hidden>
    <div class="card">
      <h2>SAM3 naming quality report</h2>
      <p id="qualitySummary"></p>
      <div class="actions">
        <a id="qualityDownload" href="/api/quality_report" download>Download saved JSON report</a>
        <button id="qualityBack" type="button">Back to settings</button>
      </div>
      <div class="quality-table-wrap">
        <table class="quality-table">
          <thead><tr><th>Segment</th><th>Word</th><th>Mean IoU</th><th>Mean SAM3 confidence</th><th>Mean score / 100</th><th>Prompt runs</th></tr></thead>
          <tbody id="qualityRows"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Game -->
  <section id="game" hidden>
    <div class="actions" style="margin:0 0 10px">
      <strong id="progress"></strong>
      <span class="spacer"></span>
      <button id="newGame" class="primary">New game (N)</button>
      <button id="toMenu">Menu</button>
    </div>
    <div id="banner" hidden></div>
    <div class="actions" style="margin:0 0 8px" aria-label="Image zoom controls">
      <button id="zoomOut" type="button" aria-label="Zoom out" title="Zoom out">-</button>
      <span id="zoomLevel" class="note" aria-live="polite">100%</span>
      <button id="zoomIn" type="button" aria-label="Zoom in" title="Zoom in">+</button>
      <button id="zoomReset" type="button">Reset zoom</button>
      <button id="clue" type="button">Clue</button>
    </div>
    <div id="imageViewport">
      <div id="stage">
        <img id="photo" alt="The photo to search" draggable="false">
        <img id="cheatLayer" class="ov" alt="" hidden>
        <div id="overlays"></div>
        <div id="clueBox" hidden aria-hidden="true"></div>
      </div>
    </div>
    <ul id="items"></ul>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
const SETTINGS_KEY = "hidden-object-game-settings-v1";
let info = null, gameId = null, polling = false, shownError = 0, game = null, toastTimer = null;
let imageZoom = 1, progressJobId = null, progressRates = {};
const MIN_IMAGE_ZOOM = 1, MAX_IMAGE_ZOOM = 3, IMAGE_ZOOM_STEP = 0.25;

async function api(url, opts) {
  const r = await fetch(url, opts);
  let data = {};
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error(data.error || ("Request failed (" + r.status + ")"));
  return data;
}
function view(name) { for (const v of ["menu", "working", "qualityReport", "game"]) $(v).hidden = (v !== name); }
function radio(name) { const el = document.querySelector('input[name="' + name + '"]:checked'); return el ? el.value : null; }
function setRadio(name, value) { const el = document.querySelector('input[name="' + name + '"][value="' + value + '"]'); if (el) el.checked = true; }
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 9000);
}
function menuError(msg) { const e = $("menuError"); e.textContent = msg || ""; e.hidden = !msg; }

/* ---------- settings ---------- */
function readSettings() {
  return {
    mode: radio("mode"), naming: radio("naming"), vlm_size: $("vlmSize").value,
    quality_check: $("qualityCheck").checked, quality_runs: Number($("qualityRuns").value),
    sam: $("sam").value, detail: Number($("detail").value),
    group_mode: radio("group_mode"),
    groups: [...document.querySelectorAll("#groupBoxes input:checked")].map(i => i.value),
    fast: $("fast").checked, half: $("half").checked, pool: Number($("pool").value),
    threshold: Number($("threshold").value), auto_top: Number($("auto_top").value),
    use_cache: $("useCache").checked, cheat: $("cheat").checked,
    clip_diagnostic: $("clipDiagnostic").checked
  };
}
function applySettings(s) {
  if (!s) return;
  if (s.mode) setRadio("mode", s.mode);
  if (s.naming) setRadio("naming", s.naming);
  if (s.group_mode) setRadio("group_mode", s.group_mode);
  if (s.vlm_size && [...$("vlmSize").options].some(o => o.value === s.vlm_size)) $("vlmSize").value = s.vlm_size;
  if (s.sam && [...$("sam").options].some(o => o.value === s.sam)) $("sam").value = s.sam;
  for (const [id, key] of [["detail", "detail"], ["pool", "pool"], ["threshold", "threshold"], ["auto_top", "auto_top"],
                            ["qualityRuns", "quality_runs"]]) {
    if (s[key] !== undefined) $(id).value = s[key];
  }
  for (const [id, key] of [["fast", "fast"], ["half", "half"], ["useCache", "use_cache"],
                            ["cheat", "cheat"], ["clipDiagnostic", "clip_diagnostic"],
                            ["qualityCheck", "quality_check"]]) {
    if (s[key] !== undefined) $(id).checked = !!s[key];
  }
  const groups = new Set(s.groups || ["general"]);
  document.querySelectorAll("#groupBoxes input").forEach(i => { i.checked = groups.has(i.value); });
}
function saveSettings() { try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(readSettings())); } catch (e) { /* private mode */ } }
function loadSaved() { try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)); } catch (e) { return null; } }

function setOff(box, off) {
  box.classList.toggle("off", off);
  box.querySelectorAll("input,select").forEach(e => { e.disabled = off; });
}
function updateStates() {
  const free = radio("mode") === "free";
  const vlm = radio("naming") === "vlm";
  const usesGroups = !free || radio("naming") === "clip";
  setOff($("freeBox"), !free);
  setOff($("vlmSizeBox"), !free || !vlm);
  setOff($("clipDiagnosticBox"), !free || !vlm);
  setOff($("qualityBox"), !free);
  setOff($("qualityRunsBox"), !free || !$("qualityCheck").checked);
  setOff($("speedBox"), free);
  setOff($("groupBox"), !usesGroups);
  $("start").textContent = free && $("qualityCheck").checked ? "Run SAM3 quality check" : "Start game";
  if (usesGroups) {
    $("auto_top").disabled = radio("group_mode") !== "auto";
    document.querySelectorAll("#groupBoxes input").forEach(i => { i.disabled = radio("group_mode") !== "custom"; });
  }
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.ceil(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60), remaining = total % 60;
  if (minutes < 60) return `${minutes}m ${remaining}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function estimateStepEta(s, stage, current, total) {
  if (progressJobId !== s.job_id) {
    progressJobId = s.job_id;
    progressRates = {};
  }
  const key = `${stage}:${total}`;
  const now = performance.now();
  const previous = progressRates[key];
  if (!previous || current < previous.current) {
    progressRates[key] = { current, observedAt: now, secondsPerItem: null };
    return null;
  }
  if (current > previous.current) {
    const sample = (now - previous.observedAt) / 1000 / (current - previous.current);
    previous.secondsPerItem = previous.secondsPerItem === null
      ? sample
      : previous.secondsPerItem * 0.7 + sample * 0.3;
    previous.current = current;
    previous.observedAt = now;
  }
  return previous.secondsPerItem === null
    ? null
    : previous.secondsPerItem * Math.max(0, total - current);
}

function updateJobProgress(s) {
  const text = $("jobProgressText");
  const bar = $("jobProgressBar");
  const log = s.log || "";
  const elapsed = formatDuration(s.elapsed_seconds || 0);
  const qualityStarted = /Starting SAM3 quality check:/.test(log);
  const qualityStep = [...log.matchAll(/Quality run (\d+)\/(\d+): word (\d+)\/(\d+) complete - (.+)/g)].pop();
  const searched = [...log.matchAll(/searched (\d+)\/(\d+) concepts/g)].pop();
  const namingMarker = [...log.matchAll(/Naming (\d+) segments\.\.\./g)].pop();
  const namingLog = namingMarker
    ? log.slice(namingMarker.index + namingMarker[0].length)
    : "";
  const named = [...namingLog.matchAll(/^\s*(\d+)\/(\d+):/gm)].pop();

  if (qualityStep) {
    const run = Number(qualityStep[1]), runs = Number(qualityStep[2]);
    const word = Number(qualityStep[3]), words = Number(qualityStep[4]);
    const current = (run - 1) * words + word, total = runs * words;
    const eta = estimateStepEta(s, "quality", current, total);
    const etaText = eta === null ? "estimating ETA..." : `step ETA ~${formatDuration(eta)}`;
    text.textContent = `SAM3 quality check: ${current} of ${total} prompts (run ${run}/${runs}, ${qualityStep[5]}) | elapsed ${elapsed} | ${etaText}`;
    bar.value = total ? Math.round(current / total * 100) : 0;
    return;
  }
  if (qualityStarted) {
    bar.removeAttribute("value");
    text.textContent = `Starting SAM3 quality check | elapsed ${elapsed} | estimating ETA...`;
    return;
  }
  if (searched) {
    const current = Number(searched[1]), total = Number(searched[2]);
    const eta = estimateStepEta(s, "search", current, total);
    const etaText = eta === null ? "estimating ETA..." : `step ETA ~${formatDuration(eta)}`;
    text.textContent = `Searching for objects: ${current} of ${total} concepts | elapsed ${elapsed} | ${etaText}`;
    bar.value = total ? Math.round(current / total * 100) : 0;
    return;
  }
  if (named) {
    const current = Number(named[1]), total = Number(named[2]);
    const eta = estimateStepEta(s, "naming", current, total);
    const etaText = eta === null ? "estimating ETA..." : `step ETA ~${formatDuration(eta)}`;
    text.textContent = `Naming objects: ${current} of ${total} | elapsed ${elapsed} | ${etaText}`;
    bar.value = total ? Math.round(current / total * 100) : 0;
    return;
  }

  bar.removeAttribute("value");
  if (namingMarker) {
    text.textContent = `Naming objects from ${namingMarker[1]} image segments | elapsed ${elapsed} | estimating ETA...`;
  } else if (/Segmenting with /.test(log)) {
    text.textContent = `Finding objects in the photo | elapsed ${elapsed} | ETA unavailable during segmentation`;
  } else if (/Recognising the scene/.test(log)) {
    text.textContent = `Recognising the scene | elapsed ${elapsed}`;
  } else if (/Loading (?:the naming model|CLIP|facebook\/)/i.test(log)) {
    text.textContent = `Loading an AI model | elapsed ${elapsed}`;
  } else if (/Image encoded once|Searching for \d+ concepts/.test(log)) {
    text.textContent = `Searching the photo for objects | elapsed ${elapsed}`;
  } else if (/Choosing and naming/.test(log)) {
    text.textContent = `Choosing objects for the game | elapsed ${elapsed}`;
  } else if (/Loaded \d+ segments from cache/.test(log)) {
    text.textContent = `Preparing saved image segments | elapsed ${elapsed}`;
  } else if (/kept \d+ segments/.test(log)) {
    text.textContent = `Preparing image segments | elapsed ${elapsed}`;
  } else if (s.kind === "newgame") {
    text.textContent = `Choosing objects for the next game | elapsed ${elapsed}`;
  } else {
    text.textContent = `Preparing the game | elapsed ${elapsed}`;
  }
}

function renderQualityReport(report, reportPath) {
  const rows = $("qualityRows");
  rows.textContent = "";
  for (const item of report.items) {
    const row = document.createElement("tr");
    const values = [
      item.segment,
      item.word,
      (item.mean_iou * 100).toFixed(1) + "%",
      (item.mean_sam3_confidence * 100).toFixed(1) + "%",
      item.mean_score.toFixed(1)
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    const detailsCell = document.createElement("td");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Details";
    details.appendChild(summary);
    for (const run of item.runs) {
      const line = document.createElement("div");
      line.textContent = `Run ${run.run}: "${run.prompt}" | IoU ${(run.iou * 100).toFixed(1)}% | confidence ${(run.sam3_confidence * 100).toFixed(1)}% | score ${run.score.toFixed(1)}`;
      details.appendChild(line);
    }
    detailsCell.appendChild(details);
    row.appendChild(detailsCell);
    rows.appendChild(row);
  }
  $("qualitySummary").textContent =
    `Evaluated ${report.items.length} named segments over ${report.runs} prompt runs. ` +
    `${report.score_definition} Saved to ${reportPath}.`;
  $("qualityDownload").href = "/api/quality_report";
}

/* ---------- start-up ---------- */
async function loadInfo() {
  try { info = await api("/api/info"); }
  catch (e) {
    $("startupText").textContent = "Connecting to the game server...";
    setTimeout(loadInfo, 1500);
    return;
  }
  if (info.phone_url) {
    $("phoneUrl").href = info.phone_url;
    $("phoneUrl").textContent = info.phone_url;
    $("phoneAccess").hidden = false;
  }
  if (!info.ready) {
    $("deviceNote").textContent = info.load_error
      ? "Could not load the libraries: " + info.load_error
      : "Loading application libraries...";
    $("startupText").textContent = info.load_error
      ? "The application libraries could not be loaded."
      : "Loading application libraries...";
    $("startupBar").hidden = !!info.load_error;
    if (!info.load_error) setTimeout(loadInfo, 1000);
    return;
  }
  for (const label of info.sam_choices) { const o = document.createElement("option"); o.value = o.textContent = label; $("sam").appendChild(o); }
  for (const g of info.groups) {
    const l = document.createElement("label");
    l.style.cssText = "display:inline-block;margin-right:14px";
    l.innerHTML = '<input type="checkbox" value="' + g + '"> ' + g;
    $("groupBoxes").appendChild(l);
  }
  applySettings(loadSaved());
  updateStates();
  $("startupProgress").hidden = true;
  $("deviceNote").textContent = info.device_label;
  $("start").disabled = false;
}

/* ---------- starting a game and following progress ---------- */
async function start() {
  menuError("");
  const file = $("photoFile").files[0];
  if (!file) { menuError("Please choose a photo (PNG or JPEG) first."); return; }
  const s = readSettings();
  saveSettings();
  const fd = new FormData();
  fd.append("photo", file);
  fd.append("settings", JSON.stringify(s));
  $("start").disabled = true;
  try { await api("/api/start", { method: "POST", body: fd }); }
  catch (e) { menuError(e.message); $("start").disabled = false; return; }
  $("start").disabled = false;
  beginPolling();
}
function beginPolling() { if (!polling) { polling = true; poll(); } }
async function poll() {
  let s;
  try { s = await api("/api/status"); } catch (e) { setTimeout(poll, 1500); return; }
  if (s.error_id > shownError) { shownError = s.error_id; toast(s.error); menuError(s.error); }
  if (s.phase === "working") {
    $("workTitle").textContent = s.kind === "newgame" ? "Choosing and naming objects\u2026"
      : s.kind === "quality" ? "Checking names with SAM3\u2026"
      : "Finding the objects\u2026";
    $("workNote").textContent = s.kind === "quality"
      ? "The interactive game is skipped; a JSON report will be saved."
      : "The game opens automatically when it is ready.";
    updateJobProgress(s);
    view("working");
    setTimeout(poll, 600);
    return;
  }
  polling = false;
  if (s.phase === "report") {
    renderQualityReport(s.quality_report, s.quality_report_path);
    view("qualityReport");
  } else if (s.phase === "ready") {
    if (s.game_id !== gameId) await loadGame();
    view("game");
  } else {
    view("menu");
  }
}
async function cancel() { try { await api("/api/cancel", { method: "POST" }); } catch (e) { /* ignore */ } }

/* ---------- the game ---------- */
function addOverlay(i) {
  const img = document.createElement("img");
  img.className = "ov"; img.alt = ""; img.src = "/api/overlay/" + gameId + "/" + i + ".png";
  $("overlays").appendChild(img);
}
function updateProgress(found, total) {
  $("clue").disabled = total === 0 || found === total;
  const banner = $("banner");
  if (total === 0) {
    $("progress").textContent = "No objects to find";
    banner.textContent = "No objects were found. Press N to try again, or go back to the menu and change the settings.";
    banner.hidden = false;
  } else if (found === total) {
    $("progress").textContent = "Found " + found + " of " + total;
    banner.textContent = "All " + total + " objects found! Press N for a new game.";
    banner.hidden = false;
  } else {
    $("progress").textContent = "Found " + found + " of " + total + ": click the objects in the picture";
    banner.hidden = true;
  }
}
async function loadGame() {
  game = await api("/api/game");
  gameId = game.game_id;
  $("clueBox").hidden = true;
  setImageZoom(1);
  $("imageViewport").scrollTo(0, 0);
  $("photo").src = "/api/image?g=" + gameId;
  $("overlays").innerHTML = "";
  $("cheatLayer").hidden = !game.cheat;
  if (game.cheat) $("cheatLayer").src = "/api/cheat/" + gameId + ".png";
  const ul = $("items");
  ul.innerHTML = "";
  game.items.forEach((label, i) => {
    const li = document.createElement("li");
    li.id = "item" + i; li.textContent = "\u2022 " + label;
    if (game.found.includes(i)) li.classList.add("found");
    ul.appendChild(li);
  });
  game.found.forEach(addOverlay);
  updateProgress(game.found.length, game.items.length);
  $("resume").hidden = false;
}
function setImageZoom(value) {
  imageZoom = Math.max(MIN_IMAGE_ZOOM, Math.min(MAX_IMAGE_ZOOM, value));
  $("stage").style.transform = "scale(" + imageZoom + ")";
  $("zoomLevel").textContent = Math.round(imageZoom * 100) + "%";
  $("zoomOut").disabled = imageZoom <= MIN_IMAGE_ZOOM;
  $("zoomIn").disabled = imageZoom >= MAX_IMAGE_ZOOM;
}
async function showClue() {
  const button = $("clue");
  button.disabled = true;
  try {
    const res = await api("/api/clue", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify({ game_id: gameId }) });
    const box = $("clueBox");
    if (res.clue) {
      box.style.left = (res.clue.x * 100) + "%";
      box.style.top = (res.clue.y * 100) + "%";
      box.style.width = (res.clue.width * 100) + "%";
      box.style.height = (res.clue.height * 100) + "%";
      box.hidden = false;
    } else {
      box.hidden = true;
    }
  } catch (e) { toast(e.message); }
  finally { button.disabled = game.found.length === game.items.length; }
}
async function onStageClick(ev) {
  const r = $("stage").getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width, y = (ev.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  try {
    const res = await api("/api/click", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ x: x, y: y, game_id: gameId }) });
    if (res.new) {
      addOverlay(res.hit);
      $("item" + res.hit).classList.add("found");
      game.found.push(res.hit);
      $("clueBox").hidden = true;
      updateProgress(res.found, res.total);
    }
  } catch (e) { toast(e.message); }
}
async function newGame() {
  try { await api("/api/new_game", { method: "POST" }); beginPolling(); }
  catch (e) { toast(e.message); }
}

/* ---------- wiring ---------- */
document.querySelectorAll('input[name="mode"],input[name="naming"],input[name="group_mode"]')
  .forEach(el => el.addEventListener("change", updateStates));
$("qualityCheck").addEventListener("change", updateStates);
$("qualityBack").addEventListener("click", () => view("menu"));
$("photoFile").addEventListener("change", () => {
  const f = $("photoFile").files[0], p = $("photoPreview");
  if (f) { p.src = URL.createObjectURL(f); p.hidden = false; } else { p.hidden = true; }
});
$("start").addEventListener("click", start);
$("cancel").addEventListener("click", cancel);
$("stage").addEventListener("click", onStageClick);
$("zoomIn").addEventListener("click", () => setImageZoom(imageZoom + IMAGE_ZOOM_STEP));
$("zoomOut").addEventListener("click", () => setImageZoom(imageZoom - IMAGE_ZOOM_STEP));
$("zoomReset").addEventListener("click", () => {
  setImageZoom(1);
  $("imageViewport").scrollTo(0, 0);
});
$("clue").addEventListener("click", showClue);
$("newGame").addEventListener("click", newGame);
$("toMenu").addEventListener("click", () => view("menu"));
$("resume").addEventListener("click", () => view("game"));
$("quit").addEventListener("click", async () => {
  if (!confirm("Stop the program? You can then close this tab.")) return;
  try { await api("/api/quit", { method: "POST" }); }
  catch (e) { if (!(e instanceof TypeError)) { toast(e.message); return; } }   // a network error just means the server is gone
  document.body.innerHTML = "<main><p>The program has stopped. You can close this tab.</p></main>";
});
document.addEventListener("keydown", ev => {
  if ($("game").hidden || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (ev.key === "n" || ev.key === "N") newGame();
});

if (!['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname)) $("quit").hidden = true;   // only the host computer may stop the game
loadInfo();
beginPolling();   // picks up a game in progress after a page reload
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
def lan_address():
    """This computer's address on the local network (no internet needed), or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))     # nothing is sent; this only picks the right network card
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None


def main():
    parser = argparse.ArgumentParser(description="Hidden Object Game as a local web app.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on. The default (127.0.0.1) allows only this computer; "
                             "use 0.0.0.0 to let phones and tablets on your Wi-Fi connect too")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    parser.add_argument("--training", action="store_true",
                        help="open the CLIP label-training interface instead of the game")
    cli = parser.parse_args()
    app.config["TRAINING_MODE"] = cli.training

    ip = None
    if cli.host not in ("127.0.0.1", "localhost", "::1"):
        ip = lan_address()
        if ip:
            app.config["PHONE_URL"] = f"http://{ip}:{cli.port}/"

    threading.Thread(target=SESSION.load_modules, daemon=True).start()
    url = f"http://127.0.0.1:{cli.port}/"
    if not cli.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no log line for every progress poll
    title = "CLIP Label Training" if cli.training else "Hidden Object Game"
    print(f"{title} is running at {url}")
    if cli.host not in ("127.0.0.1", "localhost", "::1"):
        print()
        print("Phones and tablets on the same Wi-Fi can play too. Open this address on the phone:")
        print(f"    http://{ip or '<this computer address>'}:{cli.port}/")
        print("Only do this on a network you trust: anyone on it can open the game and use your computer's")
        print("processor. If Windows asks, allow Python through the firewall for PRIVATE networks only.")
        print("Everyone connected shares the same game. Only this computer can stop the program.")
        print()
    print("Use the Quit button on the page (or Ctrl+C here) to stop.")
    app.run(host=cli.host, port=cli.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
