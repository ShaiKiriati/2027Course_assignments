#!/usr/bin/env python3
"""
Checks that this computer is ready to run the Hidden Object Game.

    python check_setup.py

It prints one line per check, and at the end tells you what to fix. Nothing is
changed on your computer (it only downloads one tiny file, config.json, to test
your access to the SAM 3 model).
"""
import importlib
import platform
import sys
from pathlib import Path

PROBLEMS = []     # things that stop the game from running
NOTES = []        # things worth knowing, but not blocking


def line(ok, text):
    """ok True -> OK, False -> FAIL, None -> informational."""
    print({True: "  OK    ", False: "  FAIL  ", None: "  --    "}[ok] + text)


def version_of(module_name, dist_name=None):
    """Return (version, error). Imports the module, so a broken install shows up as an error."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        from importlib.metadata import version
        return version(dist_name or module_name), None
    except Exception:
        return "installed", None


def main():
    print(f"Python {platform.python_version()} on {platform.system()} ({sys.executable})")
    ok = (3, 10) <= sys.version_info[:2] <= (3, 13)
    line(ok, "Python version (3.10 to 3.12 recommended)")
    if not ok:
        NOTES.append("Python 3.10 to 3.12 is recommended; other versions may not have the needed wheels.")

    print("\nPackages")
    for pip_name, module in [("torch", "torch"), ("torchvision", "torchvision"), ("transformers", "transformers"),
                             ("accelerate", "accelerate"), ("pillow", "PIL"), ("numpy", "numpy"),
                             ("matplotlib", "matplotlib"), ("flask", "flask"), ("huggingface_hub", "huggingface_hub")]:
        version, err = version_of(module, pip_name)
        line(version is not None, f"{pip_name}: {version if version else 'NOT INSTALLED (' + str(err)[:80] + ')'}")
        if version is None:
            if pip_name in ("torch", "torchvision"):
                PROBLEMS.append("Install the PyTorch build matching your GPU. For AMD ROCm, follow the "
                                "AMD-specific instructions in README.md.")
            else:
                PROBLEMS.append(f"Missing package {pip_name}. Install requirements.txt, or requirements-amd.txt "
                                "after the AMD ROCm wheels.")

    print("\nTransformers models")
    try:
        import transformers
        for cls in ["Sam3Model", "Sam3Processor", "Sam3Config", "CLIPModel", "AutoModelForImageTextToText"]:
            present = hasattr(transformers, cls)
            line(present, cls)
            if not present:
                PROBLEMS.append(f"This version of transformers has no {cls}. Run:  pip install -U \"transformers>=5.0,<6\"")
    except Exception:
        line(False, "transformers could not be imported, so its models were not checked")

    print("\nHardware")
    try:
        import torch
        if torch.cuda.is_available():
            backend = getattr(torch.version, "hip", None)
            vendor = "AMD GPU (ROCm)" if backend else "NVIDIA GPU"
            line(True, f"{vendor} found: {torch.cuda.get_device_name(0)} (the game will be fast)")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            line(True, "Apple GPU (mps) found")
        else:
            line(True, "No supported GPU: the game will run on the CPU (works, but is slow)")
            if getattr(torch.version, "hip", None):
                NOTES.append("ROCm PyTorch is installed but no supported AMD GPU was detected. Check the ROCm/driver "
                             "compatibility for this GPU.")
            elif getattr(torch.version, "cuda", None) is None:
                NOTES.append("This PyTorch is a CPU-only build. If you do have an NVIDIA GPU, reinstall PyTorch from "
                             "https://pytorch.org/get-started/locally/ with your CUDA version.")
    except Exception:
        line(False, "torch could not be imported, so the hardware was not checked")

    print("\nDesktop window support (only needed for hidden_objects_app.py)")
    version, _ = version_of("tkinter")
    line(True if version is not None else None,
         "tkinter" if version else "tkinter is missing (fine: the web version does not need it)")
    if version is None:
        NOTES.append("tkinter is missing: use the web version (hidden_objects_web.py), or install tkinter "
                     "(Debian/Ubuntu: sudo apt install python3-tk).")

    print("\nGame files in this folder")
    here = Path(__file__).resolve().parent
    for name in ["segment_and_name_sam3.py", "segment_then_name.py", "clip_word_groups.json",
                 "hidden_objects_web.py", "hidden_objects_app.py"]:
        exists = (here / name).is_file()
        line(exists, name)
        if not exists and name != "hidden_objects_app.py":
            PROBLEMS.append(f"{name} is missing from {here}")

    print("\nHugging Face account and SAM 3 access")
    try:
        from huggingface_hub import HfApi, hf_hub_download
        try:
            who = HfApi().whoami()
            line(True, f"logged in as {who.get('name', '?')}")
        except Exception:
            line(False, "could not confirm a Hugging Face login (not logged in, or no internet)")
            NOTES.append("Log in with:  hf auth login   (older versions: huggingface-cli login), using a token "
                         "from huggingface.co/settings/tokens.")
        try:
            hf_hub_download("facebook/sam3", "config.json")
            line(True, "access to facebook/sam3 (the word-list game and the SAM 3 option)")
        except Exception as e:
            kind = type(e).__name__
            if "Gated" in kind or "401" in str(e) or "403" in str(e):
                line(False, "no access to facebook/sam3 yet")
                NOTES.append("Request access at https://huggingface.co/facebook/sam3 and log in with the same account. "
                             "Until you are approved, use the open-ended game with a SAM 2.1 model (no approval needed).")
            else:
                line(False, f"could not reach Hugging Face ({kind}). Check your internet connection")
                NOTES.append("Could not reach huggingface.co, so SAM 3 access was not checked.")
    except Exception:
        line(False, "huggingface_hub is not installed, so the account was not checked")

    print()
    if PROBLEMS:
        print("Things to fix before you can play:")
        for p in dict.fromkeys(PROBLEMS):
            print("  - " + p)
    else:
        print("No blocking problems found.")
    if NOTES:
        print("Worth knowing:")
        for n in dict.fromkeys(NOTES):
            print("  - " + n)


if __name__ == "__main__":
    main()
