#!/usr/bin/env python3
"""
Hidden Object Game: launcher window (no command line needed).

Pick a photo, choose a few options, press Start. The program finds the objects,
then opens the game window. When you close the game, you are back at this window.

Two kinds of game:
  - word list: SAM 3 searches the photo for objects from a list of words
  - open-ended: SAM segments the photo first, random segments are picked and named

Keep this file in the same folder as segment_and_name_sam3.py and segment_then_name.py.
Start it with:   python hidden_objects_app.py
(or double-click play_hidden_objects.bat on Windows)
"""
import contextlib
import importlib
import queue
import re
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")          # same toolkit as this window
import matplotlib.pyplot as plt  # noqa: E402,F401  (import on the main thread, before any worker starts)

MODULE_NAMES = {"list": "segment_and_name_sam3", "free": "segment_then_name"}
SAM_CHOICES = {
    "SAM 3 (most accurate)": "facebook/sam3",
    "SAM 2.1 base+ (faster)": "facebook/sam2.1-hiera-base-plus",
    "SAM 2.1 small (fast)": "facebook/sam2.1-hiera-small",
    "SAM 2.1 tiny (fastest)": "facebook/sam2.1-hiera-tiny",
}
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
POLL_MS = 100


# --------------------------------------------------------------------------
# Log handling (plain Python, no Tk)
# --------------------------------------------------------------------------
class LogBuffer:
    """Collects text like a terminal does: a carriage return makes the next text
    overwrite the current line (that's how progress lines update in place)."""

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


class QueueWriter:
    """File-like object that forwards everything written to it onto a queue."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(("log", text))
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


# --------------------------------------------------------------------------
# Background work
# --------------------------------------------------------------------------
def load_core(q):
    """Import the heavy libraries (torch, transformers) without freezing the window."""
    try:
        q.put(("core", {key: importlib.import_module(name) for key, name in MODULE_NAMES.items()}))
    except BaseException:
        q.put(("core_error", traceback.format_exc()))


def worker(module, path, args, stop_event, q):
    """Find the objects in the photo (or load them from the cache)."""
    writer = QueueWriter(q)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            image, instances, extras = module.prepare(path, args, stop_event)
        q.put(("done", (image, instances, extras)))
    except module.Cancelled:
        q.put(("cancelled", None))
    except BaseException:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
class Launcher:
    def __init__(self, root, settings, modules):
        self.root = root
        self.settings = settings
        self.modules = modules                      # {"list": ..., "free": ...} once loaded
        self.core = modules["list"] if modules else None
        self.q = queue.Queue()
        self.log = LogBuffer()
        self.stop_event = threading.Event()
        self.working = False
        self.cheat_at_start = False
        self.active_module = None
        self.result = None   # (module, image, instances, cheat, extras) when a game should start
        self.group_vars = {}
        self.group_buttons = []

        s = settings
        self.path = tk.StringVar(value=s.get("path", ""))
        self.mode = tk.StringVar(value=s.get("mode", "list"))
        self.naming = tk.StringVar(value=s.get("naming", "vlm"))
        self.sam_label = tk.StringVar(value=s.get("sam_label", next(iter(SAM_CHOICES))))
        self.detail = tk.IntVar(value=s.get("detail", 24))
        self.group_mode = tk.StringVar(value=s.get("group_mode", "auto"))
        self.fast = tk.BooleanVar(value=s.get("fast", True))
        self.half = tk.BooleanVar(value=s.get("half", False))
        self.pool = tk.IntVar(value=s.get("pool", 30))
        self.auto_top = tk.IntVar(value=s.get("auto_top", 3))
        self.threshold = tk.DoubleVar(value=s.get("threshold", 0.5))
        self.use_cache = tk.BooleanVar(value=s.get("use_cache", True))
        self.cheat = tk.BooleanVar(value=s.get("cheat", False))

        self.build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        if modules is None:
            self.set_status("Loading libraries (the first time takes a little while)\u2026")
            self.start_button.state(["disabled"])
            threading.Thread(target=load_core, args=(self.q,), daemon=True).start()
        else:
            self.on_core_ready()
        if s.get("message"):
            self.log.feed(s["message"] + "\n")
            self.refresh_log()
        root.after(POLL_MS, self.poll)

    # ---- layout --------------------------------------------------------
    def build_ui(self):
        root = self.root
        root.title("Hidden Object Game")
        root.minsize(880, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1, uniform="cols")
        main.columnconfigure(1, weight=1, uniform="cols")
        main.rowconfigure(6, weight=1)

        # Photo
        row = ttk.Frame(main)
        row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Photo:").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(row, textvariable=self.path).grid(row=0, column=1, sticky="ew")
        ttk.Button(row, text="Browse\u2026", command=self.browse).grid(row=0, column=2, padx=(6, 0))

        # Left column: kind of game + naming
        mode_box = ttk.LabelFrame(main, text="What kind of game?", padding=8)
        mode_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=4)
        ttk.Radiobutton(mode_box, text="Search for objects from a word list", value="list",
                        variable=self.mode, command=self.update_states).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_box, text="Segment first, then name random segments", value="free",
                        variable=self.mode, command=self.update_states).grid(row=1, column=0, sticky="w")

        self.naming_box = ttk.LabelFrame(main, text="Open-ended mode settings", padding=8)
        self.naming_box.grid(row=2, column=0, sticky="nsew", padx=(0, 6), pady=4)
        nb = self.naming_box
        nb.columnconfigure(1, weight=1)
        self.naming_widgets = []
        for i, (value, text) in enumerate([("vlm", "Vision-language model (open-ended names, slower)"),
                                           ("clip", "CLIP with the word groups (faster)")]):
            r = ttk.Radiobutton(nb, text=text, value=value, variable=self.naming, command=self.update_states)
            r.grid(row=i, column=0, columnspan=2, sticky="w")
            self.naming_widgets.append(r)
        ttk.Label(nb, text="Segmentation model:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        sam_box = ttk.Combobox(nb, textvariable=self.sam_label, values=list(SAM_CHOICES), state="readonly", width=24)
        sam_box.grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(nb, text="Detail (points per side):").grid(row=3, column=0, sticky="w")
        detail_spin = ttk.Spinbox(nb, from_=8, to=48, width=6, textvariable=self.detail)
        detail_spin.grid(row=3, column=1, sticky="w")
        self.naming_widgets += [sam_box, detail_spin]

        # Right column: word groups + speed
        self.group_box = ttk.LabelFrame(main, text="Word groups", padding=8)
        self.group_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=4)
        gb = self.group_box
        self.group_radios = []
        for i, (value, text) in enumerate([("auto", "Recognise the scene and pick groups (recommended)"),
                                           ("all", "Every group (slowest)"),
                                           ("custom", "Only these groups:")]):
            r = ttk.Radiobutton(gb, text=text, value=value, variable=self.group_mode, command=self.update_states)
            r.grid(row=i, column=0, sticky="w")
            self.group_radios.append(r)
        self.group_frame = ttk.Frame(gb)
        self.group_frame.grid(row=3, column=0, sticky="w", padx=(20, 0), pady=(2, 0))

        self.speed_box = ttk.LabelFrame(main, text="Speed and quality", padding=8)
        self.speed_box.grid(row=2, column=1, sticky="nsew", padx=(6, 0), pady=4)
        sb = self.speed_box
        sb.columnconfigure(0, weight=1)
        fast_cb = ttk.Checkbutton(sb, text="Fast mode (lower resolution; good on a CPU)", variable=self.fast)
        fast_cb.grid(row=0, column=0, columnspan=2, sticky="w")
        half_cb = ttk.Checkbutton(sb, text="Half precision (NVIDIA graphics cards only)", variable=self.half)
        half_cb.grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(sb, text="Stop after this many objects (0 = all):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        pool_spin = ttk.Spinbox(sb, from_=0, to=500, width=6, textvariable=self.pool)
        pool_spin.grid(row=2, column=1, pady=(6, 0))
        ttk.Label(sb, text="Detection threshold:").grid(row=3, column=0, sticky="w")
        thr_spin = ttk.Spinbox(sb, from_=0.1, to=0.9, increment=0.05, width=6, format="%.2f",
                               textvariable=self.threshold)
        thr_spin.grid(row=3, column=1)
        ttk.Label(sb, text="Scene groups to add (automatic):").grid(row=4, column=0, sticky="w")
        self.auto_spin = ttk.Spinbox(sb, from_=1, to=6, width=6, textvariable=self.auto_top)
        self.auto_spin.grid(row=4, column=1)
        self.list_only_widgets = [fast_cb, half_cb, pool_spin, thr_spin]

        # Other
        other = ttk.Frame(main)
        other.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(other, text="Reuse saved results for this photo",
                        variable=self.use_cache).grid(row=0, column=0, sticky="w", padx=(0, 20))
        ttk.Checkbutton(other, text="Cheat mode (shows where the hidden objects are)",
                        variable=self.cheat).grid(row=0, column=1, sticky="w")

        # Buttons
        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=8)
        self.start_button = ttk.Button(buttons, text="Start game", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=0, column=1, padx=(0, 6))
        self.cancel_button.state(["disabled"])
        ttk.Button(buttons, text="Quit", command=self.on_close).grid(row=0, column=2)

        # Status + log
        self.status = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status, wraplength=820).grid(row=5, column=0, columnspan=2, sticky="w")
        log_frame = ttk.Frame(main)
        log_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        self.update_states()

    # ---- small helpers -------------------------------------------------
    def set_status(self, text):
        self.status.set(text)

    def refresh_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", self.log.text())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def enable(widgets, flag):
        for w in widgets:
            w.state(["!disabled"] if flag else ["disabled"])

    def update_states(self):
        """Grey out the settings that don't apply to the chosen kind of game."""
        free = self.mode.get() == "free"
        uses_groups = (not free) or self.naming.get() == "clip"   # word groups: list mode, or CLIP naming
        self.enable(self.naming_widgets, free)
        self.enable(self.list_only_widgets, not free)
        self.enable(self.group_radios, uses_groups)
        self.enable([self.auto_spin], uses_groups and self.group_mode.get() == "auto")
        self.enable(self.group_buttons, uses_groups and self.group_mode.get() == "custom")

    def browse(self):
        start_dir = str(Path(self.path.get()).parent) if self.path.get() else None
        chosen = filedialog.askopenfilename(title="Choose a photo", filetypes=IMAGE_TYPES,
                                            initialdir=start_dir)
        if chosen:
            self.path.set(chosen)

    def set_working(self, working):
        self.working = working
        self.start_button.state(["disabled"] if working else ["!disabled"])
        self.cancel_button.state(["!disabled"] if working else ["disabled"])

    def save_settings(self):
        self.settings.update(
            path=self.path.get(), mode=self.mode.get(), naming=self.naming.get(),
            sam_label=self.sam_label.get(), group_mode=self.group_mode.get(), fast=self.fast.get(),
            half=self.half.get(), use_cache=self.use_cache.get(), cheat=self.cheat.get(),
            groups=[g for g, v in self.group_vars.items() if v.get()],
        )
        for key, var in (("pool", self.pool), ("auto_top", self.auto_top), ("threshold", self.threshold),
                         ("detail", self.detail)):
            try:
                self.settings[key] = var.get()
            except tk.TclError:
                pass
        self.settings.pop("message", None)

    # ---- core ready ----------------------------------------------------
    def on_core_ready(self):
        saved = set(self.settings.get("groups", ["general"]))
        for i, name in enumerate(self.core.CONCEPT_GROUPS):
            var = tk.BooleanVar(value=name in saved)
            self.group_vars[name] = var
            button = ttk.Checkbutton(self.group_frame, text=name, variable=var)
            button.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 12))
            self.group_buttons.append(button)
        self.update_states()

        device = self.core.pick_device()
        label = {"cuda": "an NVIDIA graphics card (fast)", "mps": "the Apple graphics chip"}.get(
            device, "the CPU only (slow, so keep Fast mode on and expect some waiting)")
        self.set_status(f"Ready. The models will run on {label}.")
        self.start_button.state(["!disabled"])

    # ---- actions ---------------------------------------------------------
    def start(self):
        path = self.path.get().strip().strip('"')
        if not path or not Path(path).is_file():
            messagebox.showerror("No photo", "Please choose a photo (PNG or JPEG) first.")
            return
        mode = self.mode.get()
        free = mode == "free"
        try:
            pool = int(self.pool.get())
            auto_top = int(self.auto_top.get())
            threshold = float(self.threshold.get())
            detail = int(self.detail.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid setting", "Please enter numbers in the number boxes.")
            return
        if auto_top < 1 or (not free and (pool < 0 or not 0 < threshold < 1)):
            messagebox.showerror("Invalid setting",
                                 "Stop-after must be 0 or more, groups at least 1, and the threshold between 0 and 1.")
            return
        if free and not 8 <= detail <= 64:
            messagebox.showerror("Invalid setting", "Detail (points per side) should be between 8 and 64.")
            return

        uses_groups = (not free) or self.naming.get() == "clip"
        group_mode = self.group_mode.get()
        if uses_groups and group_mode == "custom":
            picked = [g for g, v in self.group_vars.items() if v.get()]
            if not picked:
                messagebox.showerror("No groups", "Tick at least one group, or choose another option.")
                return
            groups = ",".join(picked)
        else:
            groups = group_mode if uses_groups else "auto"   # "auto" or "all"

        module = self.modules[mode]
        args = module.build_parser().parse_args([])
        args.groups = groups
        args.auto_top = auto_top
        args.no_cache = not self.use_cache.get()
        args.overlay = self.cheat.get()
        if free:
            args.namer = self.naming.get()
            args.sam_model = SAM_CHOICES.get(self.sam_label.get(), "facebook/sam3")
            args.grid = detail
        else:
            args.fast = self.fast.get()
            args.half = self.half.get()
            args.pool = pool
            args.threshold = threshold

        self.save_settings()
        self.cheat_at_start = args.overlay
        self.active_module = module
        self.log = LogBuffer()
        self.refresh_log()
        self.stop_event = threading.Event()
        self.set_working(True)
        self.set_status("Working\u2026 the game opens by itself when it is ready.")
        threading.Thread(target=worker, args=(module, path, args, self.stop_event, self.q),
                         daemon=True).start()

    def cancel(self):
        self.stop_event.set()
        self.set_status("Cancelling\u2026 this may take a moment while the current step finishes.")

    def on_close(self):
        if self.working and not messagebox.askyesno(
                "Quit", "Work is still in progress. Quit anyway?"):
            return
        self.stop_event.set()
        self.result = None
        self.save_settings()
        self.root.quit()

    # ---- messages from the background threads ----------------------------------
    def poll(self):
        changed = False
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.feed(payload)
                    changed = True
                elif kind == "core":
                    self.modules = payload
                    self.core = payload["list"]
                    self.on_core_ready()
                elif kind == "core_error":
                    self.log.feed(payload)
                    changed = True
                    self.set_status("Could not load the libraries. See the log below.")
                    messagebox.showerror(
                        "Setup problem",
                        "Could not load segment_and_name_sam3.py, segment_then_name.py or their libraries.\n"
                        "Make sure the files are in the same folder and the conda environment is correct.\n\n"
                        + payload.strip().splitlines()[-1])
                elif kind == "done":
                    image, instances, extras = payload
                    self.result = (self.active_module, image, instances, self.cheat_at_start, extras)
                    self.root.quit()
                    return
                elif kind == "cancelled":
                    self.log.feed("\nCancelled.\n")
                    changed = True
                    self.set_working(False)
                    self.set_status("Cancelled. Change anything you like and press Start again.")
                elif kind == "error":
                    self.log.feed("\n" + payload)
                    changed = True
                    self.set_working(False)
                    self.set_status("Something went wrong. See the log below.")
                    messagebox.showerror("Something went wrong", payload.strip().splitlines()[-1])
        except queue.Empty:
            pass
        if changed:
            self.refresh_log()
        self.root.after(POLL_MS, self.poll)


# --------------------------------------------------------------------------
# Main loop: launcher window -> game window -> launcher window -> ...
# --------------------------------------------------------------------------
def run_launcher(settings, modules):
    root = tk.Tk()
    app = Launcher(root, settings, modules)
    root.mainloop()
    result, modules = app.result, app.modules
    try:
        root.destroy()
    except tk.TclError:
        pass
    return result, modules


def main():
    settings, modules = {}, None
    while True:
        result, modules = run_launcher(settings, modules)
        if result is None:
            break
        module, image, instances, cheat, extras = result
        settings["message"] = "Game closed. Press Start to play this photo again, or choose another photo."
        module.show(image, instances, cheat, **extras)
    sys.exit(0)


if __name__ == "__main__":
    main()
