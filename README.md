# Face Verification Under Real-World Conditions

A five-stage face verification pipeline (detect → align → optional quality
gate → embed → compare) built to measure how poor lighting, occlusion, pose
and age gaps affect two pre-trained embeddings — ArcFace and a weaker
dlib-based model — on LFW, CALFW and CPLFW. The project also tests two
training-free pre-processing corrections (auto-gamma for lighting,
landmark-based mirror-fill for occlusion) and includes a quality gate that
detects and fixes (or rejects) both conditions in unlabelled images. AIML339,
Victoria University of Wellington.

## Setup

Python 3.13 (developed and tested on 3.13.5).

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`dlib` (a dependency of `face-recognition`) builds from source and needs
CMake and a C++ compiler on the system; if `pip install` fails on it, install
CMake first (e.g. `brew install cmake` on macOS, `apt install cmake` on
Ubuntu).

## Datasets and pre-trained models

None of these are included in this repository — see "Ethical
Considerations" in the report for why (dataset licence terms, and InsightFace
being research-use-only). Download them yourself and place them under
`data/raw/<name>/` and `model/` as shown.

| Resource | Source | Where it goes |
|---|---|---|
| LFW | http://vis-www.cs.umass.edu/lfw/ | `data/raw/lfw/` |
| CALFW | http://whdeng.cn/CALFW/ | `data/raw/calfw/` |
| CPLFW | http://whdeng.cn/CPLFW/ | `data/raw/cplfw/` |
| YOLOv8n-face checkpoint | https://github.com/derronqi/yolov8-face | `model/yolov8n-face-derronqi.pt` |
| InsightFace `buffalo_l` (ArcFace) | https://github.com/deepinsight/insightface (model zoo); auto-downloads on first use, or `insightface-cli model.download buffalo_l` | `~/.insightface/models/` (default) |
| dlib face recognition model | installed automatically with `face-recognition` (via the `face_recognition_models` package in requirements.txt) | handled by pip, nothing to place manually |
| MediaPipe FaceLandmarker | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | `model/face_landmarker.task` |

The official LFW/CALFW/CPLFW pair-list files (`pairs_LFW.txt`,
`pairs_CALFW.txt`, `pairs_CPLFW.txt`) are distributed alongside each dataset
and go under `data/pairs/`.

## Reproducing the report

All commands assume the repo root as the working directory and the venv
activated. Full-scale ("`_bigger`") runs over all 13k+ LFW images take
hours on a CPU-only laptop (see the report's Computational Cost section).
The "Output file" column is where each script writes its output on a full
rerun (under `data2/`, not tracked in git); a static copy of each is already
checked into `results/` so the numbers can be read without rerunning anything.

| Report item | Script | Output file | Copy in this repo |
|---|---|---|---|
| Table I — clean baseline | `src/run_script/run_scorecard_bigger.py` | `data2/processed/scorecard_bigger.txt` | `results/scorecard_bigger.txt` |
| Table I — lighting degraded/corrected | `src/auto_gamma_correct.py` (correction) + scoring script | `data2/processed/scorecard_lighting_corrected.txt` | `results/scorecard_lighting_corrected.txt` |
| Table I — occlusion degraded/mirror-fixed | `src/landmark_mirror_fix.py` (correction) + scoring script | `data2/processed/scorecard_occlusion40.txt`, `scorecard_occlusion_fixed.txt` | `results/scorecard_occlusion40.txt`, `results/scorecard_occlusion_fixed.txt` |
| Table II — mixed-collection gate test | `src/build_mixdata.py` → `src/quality_gate.py` → scoring script | `data2/processed/scorecard_mixdata.txt` | `results/scorecard_mixdata.txt` |
| Table III — McNemar p-values | `src/mcnemar_test.py` | `data2/processed/mcnemar_results.csv` | `results/mcnemar_results.csv` |
| 8-seed variance check (Run Distributions) | `src/mixdata_variance.py` | `data2/processed/mixdata_variance.json` | `results/mixdata_variance.json` |
| Fig. 2 — 8-seed plot | `src/make_figures.py` (reads the file above) | `figures/seed_variance.pdf` | not copied (regenerate from the json) |
| Raw-pixel baseline | `src/raw_pixel_baseline.py` | stdout (see report text) | not applicable |
| Alignment-bug diagnosis | `src/diagnose_baseline.py` | `worst_genuine_pairs.png` (contact sheet) | not copied |

Fig. 1 (pipeline schematic) is a hand-drawn diagram, not generated from data;
there is no script for it. See `results/README.md` for what each copied
file is and which came from a since-fixed early version of a script.

### Command order, full pipeline (per dataset, e.g. `lfw_bigger`)

```bash
# 1. Detect + align raw images
python src/detect_align.py --checkpoint model/yolov8n-face-derronqi.pt \
    --input data/raw/lfw --output data2/processed/aligned/lfw_bigger

# 2. Generate degraded conditions
python src/lighting.py --input data2/processed/aligned/lfw_bigger \
    --output-root data2/processed/lighting/lfw_bigger --gamma-values 0.4 1.8 2.5
python src/occlusion.py --input data2/processed/aligned/lfw_bigger \
    --output-root data2/processed/occlusion/lfw_bigger --size-min 0.40 --size-max 0.40

# 3. Embed every condition (both models)
PYTHONPATH=src python3 src/run_script/run_embed_bigger.py \
    --processed-dir data2/processed --embeddings-dir data2/embeddings

# 4. Score everything into a scorecard
PYTHONPATH=src python3 src/run_script/run_scorecard_bigger.py \
    --embeddings-dir data2/embeddings --output data2/processed/scorecard_bigger.txt

# 5. Corrections (needs the mesh pairing table built once first)
python src/landmark_mirror_fix.py --input data2/processed/aligned/lfw_bigger \
    --build-pairs --pairs model/mesh_pairs.npy
python src/auto_gamma_correct.py --input <a gamma folder> --output <corrected folder>
python src/landmark_mirror_fix.py --input <an occlusion folder> \
    --boxes <that folder>/boxes.json --output <fixed folder>

# 6. Quality gate + mixed-condition test
python src/build_mixdata.py --dataset lfw_bigger --output data2/processed/mixdata_lfw_bigger
PYTHONPATH=src python3 src/quality_gate.py \
    --input data2/processed/mixdata_lfw_bigger --output data2/processed/mixdata_lfw_bigger_gated

# 7. Statistics
PYTHONPATH=src python3 src/mcnemar_test.py \
    --embeddings-dir data2/embeddings --output data2/processed/mcnemar_results.csv
PYTHONPATH=src python3 src/mixdata_variance.py --dataset lfw_bigger \
    --seeds v1 v2 v3 v4 v5 v6 v7 v8
```

Repeat steps 1–7 with `calfw_bigger`/`cplfw_bigger` and their raw folders for
the other two datasets. `src/raw_pixel_baseline.py` and
`src/diagnose_baseline.py` are standalone and only need the aligned crops and
pair lists.

## Project layout

- `src/` — all pipeline code (detection/alignment, degradation generators,
  corrections, quality gate, scoring, statistics).
- `src/run_script/` — entry points that chain the above over the full
  dataset set.
- `results/` — static copies of the small output files behind each report
  table/figure (see `results/README.md`), checked into git so the numbers
  are visible without rerunning the pipeline.
- `model/` — pre-trained model weights (not tracked in git; see table above).
- `data/`, `data2/` — raw/intermediate/processed images and embeddings (not
  tracked in git). `data2/` holds the full-scale (`_bigger`) runs behind the
  final report; `data/` holds earlier, smaller pilot runs.

Dataset/results root directories (`data/`, `data2/`) and the models
directory (`model/`) are passed as CLI arguments (`--input`,
`--output`, `--embeddings-dir`, etc.) on every script above rather than
hard-coded, so they can be pointed anywhere.
