# Hidden Object Game

Load a photo and the program turns it into a hidden-object game: a list of objects appears below the picture, and you click them in the photo to strike them off. Everything runs on your own computer, and your photos never leave it.

There are two kinds of game:

- **Word list:** SAM 3 searches the photo for objects from a list of words (kitchen, bathroom, toys, and so on).
- **Open-ended:** SAM segments the photo first, random segments are picked, and a vision-language model (or CLIP) names them. VLM mode defaults to SmolVLM-500M, with the smaller SmolVLM-256M available in the settings. It preserves the full photo's aspect ratio and verifies proposed names against the highlighted scene and segment crop. An optional CLIP diagnostic reports competing labels without rejecting the VLM's result.

## What you need

- Windows, macOS or Linux, with **Python 3.10 to 3.12**
- About **10 GB of free disk space** for the libraries and the downloaded models (a rough estimate; the first run downloads them)
- An internet connection for the first run
- A free **Hugging Face account** if you want SAM 3 (see below)
- A computer with an **NVIDIA graphics card** is much faster, but not required. On a CPU everything works, but the first steps take minutes.

## 1. Get the files

Put these files in one folder:

`hidden_objects_web.py`, `hidden_objects_training.html`, `hidden_objects_app.py`, `segment_and_name_sam3.py`, `segment_then_name.py`, `clip_word_groups.json`, `requirements.txt`, `environment.yml`, `check_setup.py`, and (Windows) the launcher files.

## 2. Install the dependencies

### With conda (recommended, and what the Windows `.bat` file expects)

Open an Anaconda or Miniconda prompt in the folder and run:

```
conda env create -f environment.yml
conda activate web-obj-game
```

The environment must be called `web-obj-game` to use the Windows launchers.

### Without conda

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS and Linux)
pip install -r requirements.txt
```

### NVIDIA graphics card

Install PyTorch with CUDA support **before** the step above, using the command from the selector at https://pytorch.org/get-started/locally/ (choose Pip and your CUDA version). Then run `pip install -r requirements.txt`; pip will keep your GPU build. A plain `pip install torch` on Windows installs a CPU-only build.

## 3. Get access to SAM 3 (free, but needs approval)

SAM 3 (`facebook/sam3`) is a gated model: each person has to ask for access once.

1. Create an account at https://huggingface.co and open https://huggingface.co/facebook/sam3.
2. Accept the terms and request access. Approval is manual and can take from minutes to several days.
3. Create an access token at https://huggingface.co/settings/tokens (read permission is enough).
4. Log in on your computer:
   ```
   hf auth login
   ```
   (On older versions of the library the command is `huggingface-cli login`.) Paste the token when asked. Keep your token private and never share it or put it in the files.

**While you wait for approval,** or if you are not approved: the **open-ended game** works with the SAM 2.1 models, which need no approval. Choose "SAM 2.1 small (fast)" or similar under "Open-ended mode settings". The word-list game needs SAM 3.

## 4. Check the setup

```
python check_setup.py
```

It lists what is installed, whether a GPU is found, whether you are logged in, and whether you have access to SAM 3, and it says what to fix.

## 5. Play

**Windows:** double-click `play_hidden_objects_web.bat`. The game opens in your browser without leaving a terminal window open. If the launcher cannot find the conda environment, it tells you what to edit.

To use the developer-oriented CLIP label trainer instead, double-click `train_clip_labels_web.bat`. It opens at http://127.0.0.1:5001, separately from the normal game. The trainer shows only rejected or uncertain labels from the sampled segments. Corrections are saved directly into the source group in `clip_word_groups.json`, alongside the original words, so they are available to future training and normal CLIP runs. If a proposed label occurs in multiple selected groups, choose which group's list to update.

**Any system:**
```
python hidden_objects_web.py
```
Your browser opens the game at http://127.0.0.1:5000. When started this way, keep the terminal window open while you play, and use **Quit** on the page to stop.

There is also a desktop-window version (`python hidden_objects_app.py`, or `play_hidden_objects.bat`). It needs tkinter, which comes with conda, and with Python from python.org on Windows and macOS. On Debian or Ubuntu, install it with `sudo apt install python3-tk`.

### Play on a phone or tablet

The game can run on your computer while you play on a phone or tablet on the **same Wi-Fi network**:

1. Start it with `python hidden_objects_web.py --host 0.0.0.0` (Windows: double-click `play_hidden_objects_phone.bat`, which runs without a terminal window).
2. The web page shows the address to open in the phone's browser.
3. On Windows, if a firewall window appears, allow Python on **private** networks only.

The phone's photo picker offers the camera and photo library. The computer still does all the work, so the speed is the computer's speed.

Only do this on a network you trust (not public Wi-Fi): anyone on the network can open the game. Everyone who connects shares one game, and only the computer itself can stop the program.

### How to play

Choose a photo, pick a kind of game, and press **Start game**. The page shows the current step, elapsed time, and an approximate ETA once progress can be measured. SAM's automatic mask-generation step has no intermediate progress callback, so its bar remains indeterminate until segmentation finishes. Click objects in the photo to find them: their names are struck off the list and they are darkened. Press **N** for a new game with a fresh random set of objects from the same photo.

In **Open-ended mode**, enable **Run SAM3 naming quality check instead of playing** to analyze the names and masks selected for the opening game. Choose 1–5 prompt runs; each uses a different text template for SAM3. The report lists each segment's mean mask IoU, SAM3 confidence, and a combined score (`100 × IoU × confidence`), plus the individual run results. This combined score is a comparison heuristic, not a calibrated probability. The check always uses SAM3 and requires access to its Hugging Face model, even if a different segmentation model was selected for generating the original masks. It can take a long time, especially on a CPU. Reports are saved as JSON under `.cache/sam3_hidden_objects/quality_reports` in your user folder and can also be downloaded from the report page.

Results are saved per photo, so a photo you have played before starts almost immediately. They are stored in `.cache/sam3_hidden_objects` in your user folder; delete that folder to clear them.

## Tips if it is slow

- Keep **Fast mode** on (word-list game) and lower the **Detail** setting (open-ended game).
- Use a smaller segmentation model: "SAM 2.1 small" or "tiny".
- On the word-list game, let the scene recognition pick the word groups instead of searching every group.
- Use "CLIP with the word groups" instead of the vision-language model for naming.
- In CLIP naming mode, each proposed label is checked against the segment and highlighted scene; if it disagrees, CLIP searches all word groups for an alternative.
- In the open-ended game, CLIP checks VLM-proposed names first; only uncertain CLIP results need the slower VLM verification call. The optional `--clip-diagnostic` also prints competing labels.

To run the open-ended script with the diagnostic enabled:

```
python segment_then_name.py photo.jpg --clip-diagnostic
```

## Troubleshooting

| Problem | What to do |
|---|---|
| `check_setup.py` says a package is missing | Run `pip install -r requirements.txt` inside the right environment. |
| "gated repo" or 401 or 403 error | You are not logged in, or not yet approved for SAM 3. Run `hf auth login`, check your request at https://huggingface.co/facebook/sam3, or use a SAM 2.1 model. |
| Everything runs on the CPU although you have an NVIDIA card | You have a CPU-only PyTorch. Reinstall it from https://pytorch.org/get-started/locally/. |
| `ImportError` for `Sam3Model` | Run `pip install -U "transformers>=5.0,<6"`. |
| Windows warning about symlinks | Harmless; it is switched off in the program. |
| The page does not open | Go to http://127.0.0.1:5000 yourself. If the port is busy, use `python hidden_objects_web.py --port 5050`. |
| The phone cannot open the address | Check that the phone and computer are on the same Wi-Fi (not a guest network), that you started with `--host 0.0.0.0`, and that the firewall allows Python on private networks. |
| Out of memory | Use a smaller segmentation model, or close other programs. |

## Files

| File | Purpose |
|---|---|
| `hidden_objects_web.py` | The game as a local web app (Flask) |
| `hidden_objects_training.html` | Dedicated interactive CLIP label-training page |
| `hidden_objects_app.py` | The game as a desktop window (tkinter) |
| `segment_and_name_sam3.py` | The word-list game and the game window logic |
| `segment_then_name.py` | The open-ended game |
| `clip_word_groups.json` | Built-in and training-added words, organized by CLIP group |
| `requirements.txt`, `environment.yml` | Dependencies |
| `check_setup.py` | Checks your installation |
| `play_hidden_objects_web.bat`, `play_hidden_objects_phone.bat`, `train_clip_labels_web.bat`, `play_hidden_objects.bat` | Windows launchers (normal web, web for phones on your Wi-Fi, CLIP label trainer, and desktop versions) |

The models are the property of their authors and are downloaded from Hugging Face under their own licenses. Read the license on each model page before sharing or using the game beyond personal use.
