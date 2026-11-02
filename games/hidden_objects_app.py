#!/usr/bin/env python3
"""
Hidden Object Game: launcher window (no command line needed).

Pick a photo, choose a few options, press Start. The program finds the objects
with SAM 3, then opens the game window. When you close the game, you are back
at this window.

Keep this file in the same folder as segment_and_name_sam3.py.
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

CORE_NAME = "segment_and_name_sam3"
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
        q.put(("core", importlib.import_module(CORE_NAME)))
    except BaseException:
        q.put(("core_error", traceback.format_exc()))


def worker(core, path, args, stop_event, q):
    """Find the objects in the photo (or load them from the cache)."""
    writer = QueueWriter(q)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            image, instances = core.prepare(path, args, stop_event)
        q.put(("done", (image, instances)))
    except core.Cancelled:
        q.put(("cancelled", None))
    except BaseException:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
class Launcher:
    def __init__(self, root, settings, core):
        self.root = root
        self.settings = settings
        self.core = core
        self.q = queue.Queue()
        self.log = LogBuffer()
        self.stop_event = threading.Event()
        self.working = False
        self.cheat_at_start = False
        self.result = None            # (image, instances, cheat) when a game should start
        self.group_vars = {}
        self.group_buttons = []

        s = settings
        self.path = tk.StringVar(value=s.get("path", ""))
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

        if core is None:
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
        root.minsize(660, 640)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(6, weight=1)

        # Photo
        row = ttk.Frame(main)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Photo:").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(row, textvariable=self.path).grid(row=0, column=1, sticky="ew")
        ttk.Button(row, text="Browse\u2026", command=self.browse).grid(row=0, column=2, padx=(6, 0))

        # Which concept groups
        box = ttk.LabelFrame(main, text="Which objects to look for", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=4)
        modes = [
            ("auto", "Recognise the scene and pick the relevant groups (recommended)"),
            ("all", "Search every group (slowest)"),
            ("custom", "Only these groups:"),
        ]
        for i, (value, text) in enumerate(modes):
            ttk.Radiobutton(box, text=text, value=value, variable=self.group_mode,
                            command=self.update_group_state).grid(row=i, column=0, sticky="w")
        self.group_frame = ttk.Frame(box)
        self.group_frame.grid(row=3, column=0, sticky="w", padx=(24, 0), pady=(2, 0))

        # Speed and quality
        opt = ttk.LabelFrame(main, text="Speed and quality", padding=8)
        opt.grid(row=2, column=0, sticky="ew", pady=4)
        opt.columnconfigure(0, weight=1)
        ttk.Checkbutton(opt, text="Fast mode: lower resolution, much quicker without an NVIDIA graphics card",
                        variable=self.fast).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opt, text="Half precision (NVIDIA graphics cards only)",
                        variable=self.half).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(opt, text="Stop after finding this many objects (0 = search everything):").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(opt, from_=0, to=500, width=6, textvariable=self.pool).grid(row=2, column=1, pady=(6, 0))
        ttk.Label(opt, text="Scene groups to add in automatic mode:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(opt, from_=1, to=6, width=6, textvariable=self.auto_top).grid(row=3, column=1)
        ttk.Label(opt, text="Detection threshold (lower finds more, but less certain, objects):").grid(
            row=4, column=0, sticky="w")
        ttk.Spinbox(opt, from_=0.1, to=0.9, increment=0.05, width=6, format="%.2f",
                    textvariable=self.threshold).grid(row=4, column=1)

        # Other
        other = ttk.Frame(main)
        other.grid(row=3, column=0, sticky="w", pady=4)
        ttk.Checkbutton(other, text="Reuse saved results for this photo",
                        variable=self.use_cache).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(other, text="Cheat mode (shows where the hidden objects are)",
                        variable=self.cheat).grid(row=1, column=0, sticky="w")

        # Buttons
        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, sticky="w", pady=8)
        self.start_button = ttk.Button(buttons, text="Start game", command=self.start)
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=0, column=1, padx=(0, 6))
        self.cancel_button.state(["disabled"])
        ttk.Button(buttons, text="Quit", command=self.on_close).grid(row=0, column=2)

        # Status + log
        self.status = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status, wraplength=620).grid(row=5, column=0, sticky="w")
        log_frame = ttk.Frame(main)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(6, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        self.update_group_state()

    # ---- small helpers -------------------------------------------------
    def set_status(self, text):
        self.status.set(text)

    def refresh_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", self.log.text())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def update_group_state(self):
        state = "normal" if self.group_mode.get() == "custom" else "disabled"
        for button in self.group_buttons:
            button.configure(state=state)

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
            path=self.path.get(), group_mode=self.group_mode.get(), fast=self.fast.get(),
            half=self.half.get(), use_cache=self.use_cache.get(), cheat=self.cheat.get(),
            groups=[g for g, v in self.group_vars.items() if v.get()],
        )
        for key, var in (("pool", self.pool), ("auto_top", self.auto_top), ("threshold", self.threshold)):
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
            button.grid(row=i // 4, column=i % 4, sticky="w", padx=(0, 14))
            self.group_buttons.append(button)
        self.update_group_state()

        device = self.core.pick_device()
        label = {"cuda": "an NVIDIA graphics card (fast)", "mps": "the Apple graphics chip"}.get(
            device, "the CPU only (slow, so keep Fast mode on)")
        self.set_status(f"Ready. SAM 3 will run on {label}.")
        self.start_button.state(["!disabled"])

    # ---- actions ---------------------------------------------------------
    def start(self):
        path = self.path.get().strip().strip('"')
        if not path or not Path(path).is_file():
            messagebox.showerror("No photo", "Please choose a photo (PNG or JPEG) first.")
            return
        try:
            pool = int(self.pool.get())
            auto_top = int(self.auto_top.get())
            threshold = float(self.threshold.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid setting", "Please enter numbers in the number boxes.")
            return
        if pool < 0 or auto_top < 1 or not 0 < threshold < 1:
            messagebox.showerror("Invalid setting",
                                 "Stop-after must be 0 or more, groups at least 1, and the threshold between 0 and 1.")
            return

        mode = self.group_mode.get()
        if mode == "custom":
            picked = [g for g, v in self.group_vars.items() if v.get()]
            if not picked:
                messagebox.showerror("No groups", "Tick at least one group, or choose another option.")
                return
            groups = ",".join(picked)
        else:
            groups = mode   # "auto" or "all"

        args = self.core.build_parser().parse_args([])
        args.groups = groups
        args.fast = self.fast.get()
        args.half = self.half.get()
        args.pool = pool
        args.auto_top = auto_top
        args.threshold = threshold
        args.no_cache = not self.use_cache.get()
        args.overlay = self.cheat.get()

        self.save_settings()
        self.cheat_at_start = args.overlay
        self.log = LogBuffer()
        self.refresh_log()
        self.stop_event = threading.Event()
        self.set_working(True)
        self.set_status("Working\u2026 the game opens by itself when the objects have been found.")
        threading.Thread(target=worker, args=(self.core, path, args, self.stop_event, self.q),
                         daemon=True).start()

    def cancel(self):
        self.stop_event.set()
        self.set_status("Cancelling\u2026 this may take a moment while the current step finishes.")

    def on_close(self):
        if self.working and not messagebox.askyesno(
                "Quit", "Objects are still being searched for. Quit anyway?"):
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
                    self.core = payload
                    self.on_core_ready()
                elif kind == "core_error":
                    self.log.feed(payload)
                    changed = True
                    self.set_status("Could not load the libraries. See the log below.")
                    messagebox.showerror(
                        "Setup problem",
                        "Could not load segment_and_name_sam3.py or its libraries.\n"
                        "Make sure both files are in the same folder and the conda environment is correct.\n\n"
                        + payload.strip().splitlines()[-1])
                elif kind == "done":
                    image, instances = payload
                    self.result = (image, instances, self.cheat_at_start)
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
def run_launcher(settings, core):
    root = tk.Tk()
    app = Launcher(root, settings, core)
    root.mainloop()
    result, core = app.result, app.core
    try:
        root.destroy()
    except tk.TclError:
        pass
    return result, core


def main():
    settings, core = {}, None
    while True:
        result, core = run_launcher(settings, core)
        if result is None:
            break
        image, instances, cheat = result
        settings["message"] = "Game closed. Press Start to play this photo again, or choose another photo."
        core.show(image, instances, cheat)
    sys.exit(0)


if __name__ == "__main__":
    main()
