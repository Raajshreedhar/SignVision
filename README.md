# SignVision

SignVision is a real-time American Sign Language (ASL) alphabet detector that runs natively on an Apple Silicon Mac and uses the laptop webcam. Raise a hand anywhere in view and sign: the app finds the signing hand in the full camera frame, reads the letter, and adds it to a text panel in the same window once the sign is held steady. Besides A–Z it knows SPACE, DELETE and NOTHING.

It is a two-stage [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) pipeline on PyTorch with the Apple MPS backend: a hand detector scans the whole frame, and a letter classifier reads a grayscale, contrast-enhanced crop of the chosen hand.

## Overview

- **Whole-frame detection, no fixed box.** Every mirrored webcam frame goes through a YOLO26n detector fine-tuned on one class, `hand`. Your hand can be anywhere in view.
- **Picks the signing hand and stays with it.** Each detected hand is scored by confidence × √(box area / frame area), and the best one is selected. The selected hand stays selected while it overlaps its previous box and keeps at least half the best score, so a second hand in view does not steal the selection. When the detector misses the hand for up to 3 frames, the last box is held.
- **Grayscale + CLAHE crop.** A padded square crop of the selected hand is converted to grayscale and contrast-equalized with CLAHE, so room lighting matters less. Training and the live app use the same function.
- **29 classes.** A YOLO26n-cls classifier outputs A–Z plus three control signs: SPACE adds a space, DELETE removes the last character, NOTHING writes nothing. A frame without a hand counts as NOTHING.
- **Steady output.** Class probabilities are averaged over the last 5 frames. A sign held confidently commits once, and a NOTHING (or a dropped hand) must come between any two commits.
- **HUD.** FPS, detector and classifier latency, and the device (`mps` or `cpu`) are drawn on the video.

## Run the app

The app needs the trained models in `models/`: `hand_detector.pt`, `letter_classifier.pt` and `letter_classifier.json`. They are not committed. On a fresh clone, do [Setup](#setup) and [Build the models](#build-the-models) first.

```bash
cd SignVision
.venv/bin/python -m src.app
```

- The terminal prints `Using device: mps` (`cpu` on a Mac without MPS).
- The first time, macOS asks whether your terminal app may use the camera. Allow it, then start the app again if it exited (see [Troubleshooting](#troubleshooting)).
- A window named **SignVision** shows the mirrored camera view:
  - top left, the HUD: `FPS … det … ms cls … ms mps`, with `cls -` when no hand is in view;
  - a thick **green** box on the signing hand, with the current label and its confidence above it (`<label> <confidence>%`);
  - a **yellow** box while the hand is briefly lost (up to 3 frames);
  - thin **gray** boxes on any other hands;
  - `No hand -> NOTHING` when no hand is visible;
  - a black panel at the bottom with `Text: …`, the committed text.
- Keys, with the window focused: **q** quits and **c** clears the text. On quit, the terminal prints the session's latency summary (detector, classifier and total ms as mean / p50 / p95, and the mean FPS).

The HUD's FPS covers the whole loop, including the camera read and drawing, so the webcam's frame rate caps it.

### How to sign

1. Hold a sign steady. It commits once the smoothed confidence has stayed at or above `CONFIDENCE_THRESHOLD` (0.75) for `DEBOUNCE_FRAMES` (8) frames in a row. Holding it longer adds nothing.
2. After **every** commit, including before a different letter, drop your hand out of view or sign NOTHING briefly. Each commit starts a cooldown of `COOLDOWN_FRAMES` (12) frames. After the cooldown, NOTHING must be seen for about `SMOOTH_WINDOW` + `DEBOUNCE_FRAMES` (5 + 8) frames before the next sign can commit. In practice: sign, watch the letter appear, lower your hand for about a second, then sign the next letter.
3. SPACE and DELETE follow the same rule.

Example: to write `AB`, sign A until `Text: A` appears, lower your hand for a moment, then sign B.

These values are in `src/config.py`.

## Architecture

```
 webcam frame (1280x720, mirrored)
            |
            v
 +----------------------+
 | HandDetector         |  YOLO26n, one class "hand", imgsz 512:
 | src/pipeline.py      |  every hand in the frame
 +----------------------+
            | boxes
            v
 +----------------------+
 | HandSelector         |  best score = conf x sqrt(area share),
 | src/tracking.py      |  sticky across frames, 3-frame grace
 +----------------------+
            | raw box of the signing hand
            | (no hand: the classifier is skipped and NOTHING is fed to the smoother)
            v
 +----------------------+
 | classifier_input     |  padded square crop -> gray -> CLAHE
 | src/preprocess.py    |  -> 224x224, 3 identical channels
 +----------------------+
            v
 +----------------------+
 | LetterClassifier     |  YOLO26n-cls, 29 class probabilities
 | src/pipeline.py      |
 +----------------------+
            v
 +----------------------+
 | ProbSmoother         |  mean over the last 5 frames
 | src/smoothing.py     |
 +----------------------+
            | label + confidence
            v
 +----------------------+
 | CommitStateMachine   |  debounce, cooldown,
 | src/state_machine.py |  NOTHING between any two commits
 +----------------------+
            | text
            v
 OpenCV window: boxes, label, HUD, text panel (src/app.py)
```

`SignPipeline` in `src/pipeline.py` runs the detector → selector → crop → classifier → smoother steps for one frame. `src/app.py` feeds it camera frames and passes its output to the commit state machine.

| Module | Responsibility |
| --- | --- |
| `src/config.py` | All constants: the canonical class order, paths, detector, crop, selection, smoothing, commit and camera settings |
| `src/yolo_env.py` | The only Ultralytics import point: MPS fallback variable, analytics off for this process, device choice, weight loading, shared trainer helpers |
| `src/cli.py` | Shared argparse validators for the command-line tools |
| `src/download_data.py` | Source registry, downloads (kagglehub or a direct zip), `data/sources.json` |
| `src/hand_data.py` | Builds the YOLO hand dataset `data/hand_det/` from hand-keypoints and HaGRID (or zoom-out composites) |
| `src/train_detector.py` | Trains the YOLO26 hand detector into `models/hand_detector.{pt,json}` |
| `src/class_map.py` | Maps each source's folder names to the 29 classes; `00_A` … `28_NOTHING` folder names; `decode_class_names` |
| `src/preprocess.py` | The shared square crop and gray + CLAHE classifier input |
| `src/cls_data.py` | Builds the detector-cropped classifier datasets `data/cls/` and `data/cls_test/` |
| `src/train_classifier.py` | Trains the YOLO26-cls letter classifier into `models/letter_classifier.{pt,json}` |
| `src/tracking.py` | `Detection`, box scoring, `HandSelector` |
| `src/pipeline.py` | `HandDetector`, `LetterClassifier`, `SignPipeline` |
| `src/smoothing.py` | `ProbSmoother` |
| `src/state_machine.py` | `CommitStateMachine` |
| `src/perf.py` | `LatencyTracker`: FPS and p50 / p95 latency |
| `src/evaluate.py` | Offline evaluation of both models into `models/reports/` |
| `src/benchmark.py` | Headless FPS and latency benchmark into `models/reports/benchmark_<device>.json` |
| `src/app.py` | The live webcam app |

Generated files are gitignored and never committed: everything under `data/`, the pretrained YOLO26 weights in `models/pretrained/` (downloaded on first use), the trained models `models/hand_detector.{pt,json}` and `models/letter_classifier.{pt,json}`, training runs in `models/runs/`, reports in `models/reports/`, and smoke-test output in `models/smoke/`. Kaggle downloads are cached in `~/.cache/kagglehub`.

## Requirements

- macOS on Apple Silicon. The project was built and measured on an Apple M5 with 16 GB of memory. Training and inference use the MPS backend; everything also runs on the CPU, more slowly.
- Python 3.14 (developed with 3.14.8). See [Troubleshooting](#troubleshooting) for a Python 3.12 fallback.
- A webcam for the live app.
- Free disk space for the downloads (sizes in the [data sources](#data-sources) table) and the datasets built from them, and a network connection for the first run.

The pins are in `requirements.txt`: torch 2.14.1, torchvision 0.29.1, opencv-python 5.0.0.93, numpy 2.5.3, ultralytics 8.4.174 and pytest 9.1.1, plus pillow, kagglehub and tqdm.

## Setup

```bash
git clone https://github.com/Raajshreedhar/SignVision.git
cd SignVision
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Homebrew's Python is an "externally managed" environment (PEP 668), so a system-wide `pip install` fails. The project `.venv` avoids that. Every command in this README calls the venv's interpreter directly (`.venv/bin/python`), so you never need to activate it.

## Data sources

| Source | Role | Size | License | On the build machine |
| --- | --- | --- | --- | --- |
| [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle) | Classifier training, all 29 classes, plus its 28 test photos for the held-out test set | 1.1 GB | GPL-2 | downloaded, required |
| [debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) (Kaggle) | Classifier training | 4.6 GB | CC0 | downloaded |
| [ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset) (Kaggle) | Classifier training (letters only, digits skipped) | 32 MB | CC0 | downloaded |
| [danrasband/asl-alphabet-test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) (Kaggle) | Held-out classifier test set, never trained on | 13 MB | CC0 | downloaded |
| [hand-keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) (Ultralytics, direct zip) | Hand detector | 387 MB | CC BY-NC-SA 4.0 | downloaded, required |
| [innominate817/hagrid-sample-30k-384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p) (Kaggle) | Hand detector (webcam-style frames) and NOTHING background crops | 1.0 GB | CC BY-SA 4.0 | downloaded |

`.venv/bin/python -m src.download_data --list` prints this registry and whether each source is cached.

What each source contributed, from `data/hand_det/stats.json` and `data/cls/stats.json`:

- **Hand detector data** (`data/hand_det/`): hand-keypoints 18,720 train and 7,965 val images; HaGRID 28,518 train and 3,315 val images (34,802 and 4,082 hand boxes).
- **Classifier data** (`data/cls/`, crops made by the trained detector): grassknoted 43,500 crops (1,500 per class), debashishsau 40,434, ayuraj 1,815, and 1,500 HaGRID background crops for NOTHING. That is 78,524 train and 8,725 val crops in total.
- **Held-out test set** (`data/cls_test/`): 898 images, danrasband 870 and grassknoted 28.
- **Duplicates and leaks removed**: 171,901 debashishsau files were byte-identical copies of other training images. Training files identical to a test image were dropped as well: 28 from grassknoted and 925 from debashishsau.

### Kaggle credentials

Five sources come from Kaggle through kagglehub. On the build machine all five downloaded anonymously, without credentials. If a Kaggle download fails, `src.download_data` marks the source `skipped`, prints a hint, and carries on; only grassknoted and hand_keypoints are required. To get a skipped source:

1. On kaggle.com go to Settings → API → **Create New Token**. This downloads `kaggle.json`.
2. Either save it as `~/.kaggle/kaggle.json`:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   or set the two environment variables from it:

   ```bash
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_api_key"
   ```

3. Download the skipped source again, for example `.venv/bin/python -m src.download_data --sources hagrid30k`, then rebuild the dataset that uses it (`src.hand_data` for hagrid30k, `src.cls_data` for the classifier sources) and retrain.

### Camera-wide detector data: HaGRID or zoom-out composites

hand-keypoints images are mostly close-ups of hands. HaGRID shows people in front of a camera with their hands anywhere in the frame, which is how the app sees you. If hagrid30k is unavailable, `src.hand_data` generates zoom-out composites instead (`ZOOMOUT_COMPOSITES_TRAIN` = 8,000 and `ZOOMOUT_COMPOSITES_VAL` = 800): hand-keypoints images are shrunk and pasted at random spots on wide backgrounds (grassknoted `nothing` photos or noisy gradients), with their boxes moved to match. In that case it prints a WARNING and `data/hand_det/stats.json` records `"webcam_view_source": false`. With HaGRID the value is `true`. On the build machine HaGRID was included, so `webcam_view_source` is `true` and no composites were made.

## Build the models

The full pipeline, in order:

```bash
# 1. Download the six sources (re-runs reuse the cache; --sources a,b picks some)
.venv/bin/python -m src.download_data
# 2. Build the YOLO hand dataset in data/hand_det/
.venv/bin/python -m src.hand_data
# 3. Train the hand detector (--time is a wall-clock budget in hours)
.venv/bin/python -m src.train_detector --model n --time 2.0 --imgsz 512 --batch 16 --patience 10 --device mps
# 4. Crop every classifier image with the trained detector into data/cls/ and data/cls_test/
.venv/bin/python -m src.cls_data --device mps
# 5. Train the letter classifier
.venv/bin/python -m src.train_classifier --model n --time 1.5 --imgsz 224 --batch 64 --patience 5 --device mps
# 6. Evaluate both models (reports in models/reports/)
.venv/bin/python -m src.evaluate --split all --detector --device mps
# 7. Measure FPS and latency without a camera
.venv/bin/python -m src.benchmark --mode finetuned --device both
# 8. Start the app
.venv/bin/python -m src.app
```

- **Time budget.** With `--time` above 0, Ultralytics fits the number of epochs to the budget. `--time 0` uses `--epochs` instead.
- **Smoke tests.** `--smoke-test` trains 1 epoch on synthetic images into `models/smoke/` through the same save and validation steps, and never touches the real models:

  ```bash
  .venv/bin/python -m src.train_detector --smoke-test
  .venv/bin/python -m src.train_classifier --smoke-test
  ```

- **Resume an interrupted run** from `models/runs/<name>/weights/last.pt`: `.venv/bin/python -m src.train_detector --model n --resume` (same flag for `src.train_classifier`).
- **Continue training from the best weights.** Both measured runs ended on their time budget while the validation metrics were still improving, so a longer run may help:

  ```bash
  .venv/bin/python -m src.train_detector --model n --base models/hand_detector.pt --time 1.0
  .venv/bin/python -m src.train_classifier --model n --base models/letter_classifier.pt --time 1.0
  ```

- **After retraining the detector**, rebuild `data/cls` with `src.cls_data` and retrain the classifier: the classifier's crops come from the detector, and the app warns when the two do not match.
- **On the CPU**, pass `--device cpu`. If a run fails on MPS, the trainer prints the exact command to run on the CPU.
- `src.cls_data --no-detector` is a degraded mode that uses uncropped center squares, for when no trained detector exists.

`src.evaluate` writes, per split, top-1 / top-5, per-class accuracy, a 29 × 29 confusion matrix with its heatmap, the 5 weakest classes and the 5 most-confused pairs (`models/reports/classifier_<split>.json`, `confusion_<split>.png`). With `--detector` it adds Ultralytics validation of the detector, overall and per source (`models/reports/detector.json`, `models/reports/detector_val*/`). `src.benchmark --mode pretrained --matrix` times the four detector × classifier size combinations (n or s each) with the pretrained weights, `--no-force-box` uses the real detector boxes instead of a fixed box, and every run is merged into `models/reports/benchmark_<device>.json`.

## Measured results

The measured values come from the JSON files named above each table. The `--time` budgets are the training commands from [Build the models](#build-the-models).

### Hand detector

`models/hand_detector.json`, `models/reports/detector.json`

| | |
| --- | --- |
| Model | YOLO26n, fine-tuned from `yolo26n.pt`, imgsz 512 |
| Training | 6 epochs, 7,480.5 s on MPS (`--time 2.0`) |

| Val split | Images | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| hand-keypoints | 7,965 | 0.9912 | 0.8609 | 0.9733 | 0.9739 |
| HaGRID | 3,315 | 0.9947 | 0.8358 | 0.9936 | 0.9866 |
| all | 11,280 | 0.9926 | 0.8558 | 0.9807 | 0.9768 |

`cls_test_detection_rate` is 0.9655 = 1 − 31 / 898: in 31 of the 898 held-out classifier test images, the detector found no box around a letter, SPACE or DELETE sign, so `src.cls_data` cropped a center square instead.

### Letter classifier

`models/letter_classifier.json`, `models/reports/classifier_val.json`, `models/reports/classifier_test.json`

| | |
| --- | --- |
| Model | YOLO26n-cls, fine-tuned from `yolo26n-cls.pt`, imgsz 224 |
| Training | 5 epochs, 5,519.3 s on MPS (`--time 1.5`) |
| Val, 8,725 crops | top-1 0.9968, top-5 0.9993 |
| Held-out test, 898 images | top-1 0.9443, top-5 0.9844 |
| 5 weakest test classes | DELETE 0.567 (n=30), NOTHING 0.613 (n=31), J 0.742 (n=31), M 0.871 (n=31), A 0.935 (n=31) |
| Most confused on test (true → predicted, count) | J → I 7, DELETE → C 5, M → N 4, DELETE → Q 4, NOTHING → C 3 |
| 5 weakest val classes | X 0.987, T 0.990, N 0.990, E 0.993, M 0.993 |

The val crops come from the same continuous video captures as the training crops, so val accuracy is optimistic. The held-out test set has other people and backgrounds and is the realistic number.

### Speed

`models/reports/benchmark_mps.json`, `models/reports/benchmark_cpu.json`

Each run times `SignPipeline.process` on a synthetic 1280 × 720 frame with the grassknoted `A_test.jpg` pasted in: 10 warm-up frames, then 200 timed frames. FPS = 1000 / mean total ms; the camera read and drawing are not included. With the forced box, the detector still runs but the classifier always gets a fixed box, so both stages run on every frame.

| Device | Models | FPS | det ms mean / p50 / p95 | cls ms mean / p50 / p95 | total ms mean / p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| MPS | trained n + n, forced box | 90.21 | 6.41 / 6.06 / 8.81 | 4.60 / 4.43 / 5.66 | 11.09 / 10.82 / 13.40 |
| MPS | trained n + n, real detector boxes | 93.39 | 6.71 / 6.48 / 8.72 | 3.94 / 3.76 / 5.08 | 10.71 / 10.52 / 12.93 |
| CPU | trained n + n, forced box | 57.58 | 12.66 / 12.18 / 16.11 | 4.64 / 4.49 / 5.20 | 17.37 / 16.79 / 21.53 |
| MPS | pretrained det n + cls n | 96.96 | 5.84 / 5.24 / 10.62 | 4.41 / 4.17 / 5.76 | 10.31 / 9.50 / 17.27 |
| MPS | pretrained det n + cls s | 59.71 | 8.25 / 8.14 / 8.76 | 8.43 / 8.44 / 8.91 | 16.75 / 16.66 / 17.50 |
| MPS | pretrained det s + cls n | 53.23 | 13.55 / 14.61 / 15.72 | 5.18 / 5.35 / 6.34 | 18.79 / 20.24 / 21.83 |
| MPS | pretrained det s + cls s | 55.09 | 11.98 / 11.93 / 13.91 | 6.11 / 6.05 / 7.00 | 18.15 / 17.89 / 20.96 |

Every run is above `FPS_TARGET` (25). MPS timings vary from run to run on the same machine. The pretrained rows measure speed only: those models were not trained on hands or letters.

**Sizes chosen:** the YOLO26n detector at imgsz 512 and the YOLO26n-cls classifier at imgsz 224. The n models met the project's accuracy targets and are the fastest pair in the matrix above, so the larger s models and a smaller detector input size were not needed.

## Tests

```bash
.venv/bin/python -m pytest -m "not slow" -q   # quick suite
.venv/bin/python -m pytest -q                 # full suite, including the slow trainer round trips
```

- A fixture makes `cv2.VideoCapture` raise in every test, so no test can open the camera.
- Tests that need the trained models or the downloaded data are marked `requires_weights` / `requires_data` and skip with a reason when those are missing. The slow trainer round trips (`tests/test_train_smoke.py`) download the pretrained YOLO26 weights on first use.
- The suite covers the downloads, both dataset builders, the class mapping, crop geometry and preprocessing, the hand selector, smoother, pipeline and commit rules, the app overlay and its missing-weights exit, the benchmark, the evaluator and the trainer guards. It also runs a real 1-epoch train / save / reload round trip for both models and one real-weights end-to-end frame.

## How the parts stay consistent

- **One preprocessing function.** `preprocess.classifier_input(frame, box, crop_pad, imgsz)` makes the padded square crop (side = longest box side × (1 + 2 × `CROP_PAD`), shifted to stay inside the frame), converts it to grayscale, applies CLAHE and resizes it to 224 × 224 with three identical channels. `src.cls_data` uses it for every training and test crop, and the live pipeline uses it on every frame. A test checks that `createCLAHE` and `COLOR_BGR2GRAY` appear nowhere else in `src/`.
- **Class order.** `config.CLASSES` is A–Z, then SPACE, DELETE, NOTHING. The class folders are named `00_A` … `28_NOTHING`, so the sorted folder order that Ultralytics turns into class indices is the canonical order. `class_map.decode_class_names` checks the trained model's class names when the classifier is saved and again when it is loaded, and `models/letter_classifier.json` stores the list.
- **Crops from the raw detection.** The live crop uses the detector's raw box for the selected hand, the same kind of box the training crops were made from. The smoothed box is used only for drawing and for following the hand.
- **Geometry chain.** `data/cls/stats.json` records `crop_pad`, `cls_imgsz`, `preprocess_version` and the md5 of the detector that made the crops. `src.train_classifier` refuses a dataset that does not match the code and copies these values into `models/letter_classifier.json`. The app takes the crop settings from that file, and the app and the evaluator refuse a classifier whose `preprocess_version` differs from the code's. The app prints a WARNING when `models/hand_detector.pt` is not the detector the crops were made with.

## Licenses

**AGPL-3.0 note.** Ultralytics and the YOLO26 weights are licensed under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). Anyone who distributes this app or a derivative of it, or lets others use it over a network, must release the complete source code under AGPL-3.0 or obtain an [Ultralytics Enterprise license](https://www.ultralytics.com/license).

The datasets keep their own licenses. No dataset content and no weights are committed to this repository.

| Dataset | License |
| --- | --- |
| hand-keypoints | CC BY-NC-SA 4.0 (non-commercial) |
| HaGRID (hagrid-sample-30k-384p) | CC BY-SA 4.0 |
| grassknoted/asl-alphabet | GPL-2 |
| debashishsau, ayuraj, danrasband | CC0 |

**Non-commercial use only, as trained.** The hand detector learned from hand-keypoints (CC BY-NC-SA 4.0), so treat the trained models and the project as non-commercial unless you retrain the detector without that source.

## Possible future source

Roboflow's "American Sign Language Letters" dataset could add more varied letter images. It needs a `ROBOFLOW_API_KEY` and ships a YOLO detection export rather than class folders, so it would need a converter. It is not wired in.

## Troubleshooting

- **Python 3.12 fallback.** The project was built on Python 3.14.8. If torch or ultralytics fail to install or import on your Python, make a Python 3.12 environment with the same pins and use `.venv312/bin/python` wherever this README says `.venv/bin/python` (`.venv312/` is gitignored):

  ```bash
  brew install python@3.12    # if /opt/homebrew/bin/python3.12 is missing
  /opt/homebrew/bin/python3.12 -m venv .venv312
  .venv312/bin/pip install -r requirements.txt
  ```

- **Camera permission.** If the app exits with `could not open the webcam`, allow camera access for the app you start it from (Terminal, iTerm, VS Code, …) in System Settings → Privacy & Security → Camera, quit and reopen that app, and start SignVision again. Also close other apps that use the camera. `CAMERA_INDEX` in `src/config.py` selects another camera.
- **MPS errors.** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically, so operations without an MPS kernel run on the CPU. If training fails on MPS (a runtime error, or a NaN or infinite loss or metric in the last epoch), the trainer exits with status 1 and prints the same command with `--device cpu`; run that. If the app's MPS warm-up fails, it switches both models to the CPU with one WARNING, and the HUD shows `cpu`.
- **Missing weights.** The app exits with a message naming the command to run, for example `No hand detector at models/hand_detector.pt. Train it: .venv/bin/python -m src.hand_data && .venv/bin/python -m src.train_detector`. For the classifier it names `.venv/bin/python -m src.cls_data && .venv/bin/python -m src.train_classifier`. `src.evaluate` and `src.benchmark` do the same.
- **"retrain the classifier" or "rebuild with python -m src.cls_data".** The classifier or its dataset was made by a different version of the code (class order, preprocessing version or crop settings). Rebuild `data/cls` and retrain the classifier.
- **WARNING about a different detector.** The detector was retrained after `data/cls` was built. The app still runs; rebuild with `src.cls_data` and retrain the classifier for the best accuracy.
- **A Kaggle source was skipped.** Add a Kaggle token as described in [Kaggle credentials](#kaggle-credentials).
- **`pip install` fails with "externally-managed-environment".** You used the system pip. Use `.venv/bin/pip` (see [Setup](#setup)).
- **Legacy files.** `models/signvision_cnn.pt` and `models/dataset_path.txt` belong to the earlier SignVisionCNN version with the fixed ROI box. Nothing uses them any more, and you can delete them.
