# TP2 — Image-to-Prompt Inversion (Generative AI 2025/2026)

Recover text prompts that, when re-rendered by a **fixed** Latent Consistency Model
(`SimianLuo/LCM_Dreamshaper_v7`), reproduce six target images. A vision–language
model (Google Gemini) seeds candidate prompts from each target and then iteratively
refines them as a critic, guided by the actual LCM renders. Candidates are ranked by
a composite of three image-side metrics (CLIP image–image similarity, LPIPS, pixel
RMSE) and the top-3 per target are returned.

Authors: Bernardo Ventura, Lucas Vilar.

---

## 1. Dependencies

- **Python 3.12**
- A CUDA GPU is recommended (the reported runs used a GeForce GTX 1650, 4 GB).
  CPU also works but is much slower.
- All Python dependencies are pinned in [`requirements.txt`](requirements.txt)
  (PyTorch, Diffusers, Transformers, Accelerate, `lpips`, `google-genai`, Pillow,
  NumPy, …).

Create and populate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### GPU note (GTX 16-series)
On GeForce GTX 1650/1660 the default `float16` path produces NaN / all-black renders
(a known fp16 hardware bug), so inference uses **`bfloat16`** instead. This is an
implementation-level numerical difference allowed by the assignment. On a Colab
Tesla T4, `float16` works and is faster.

---

## 2. API key

The vision–language seeder/critic uses the Google Gemini API
(`gemini-3.1-flash-lite`, free tier). Create a file named `.env` in the project
root with your key:

```
GEMINI_API_KEY=your_key_here
```

It is loaded automatically at import time (`tp2.config.load_dotenv`). `.env` is
git-ignored and **must not be committed**. A free key can be obtained at
<https://aistudio.google.com/apikey>.

---

## 3. Expected folder structure

```
.
├── tp2-chosen/                 # the 6 target images (input)
│   ├── 1159_25.png  1159_29.png  1159_3.png
│   └── 1159_7.png   7836.png     9338.png
├── tp2/                        # the method (importable package)
│   ├── config.py               # fixed LCM contract, paths, .env loader
│   ├── render.py               # LCM pipeline + render helpers
│   ├── metrics.py              # CLIP / LPIPS / pixel-RMSE evaluator
│   ├── ranking.py              # composite score + top-k selection
│   ├── targets.py              # seed-from-filename, image loading
│   ├── orchestrator.py         # run_pipeline: seed → render → score → rank → save
│   ├── io_utils.py             # per-run output writing (CSV/JSON/PNG)
│   ├── aggregate.py            # mean/std aggregation across targets
│   └── approaches/
│       ├── base.py             # trivial baselines (pipeline sanity checks)
│       └── refine.py           # Gemini seeder + render-guided critic (final method)
├── run_final.py                # entry point: full 6-target, 3-seed run
├── requirements.txt
├── .env                        # GEMINI_API_KEY (not committed)
└── outputs/                    # results are written here (created on first run)
```

---

## 4. Running the method

With the venv active, the key set in `.env`, and the targets in `tp2-chosen/`:

```bash
python run_final.py
```

This runs the full pipeline over all six targets with **3 optimiser-seed
repetitions** (the configuration reported in the paper): for each target the Gemini
seeder proposes 5 diverse prompts, then a 2-round beam search (beam width 3, 2
proposals per beam member) refines them using the LCM renders as feedback. It prints
the run directory, total time, and the top-3 prompts per target with their metrics.

> Reproducibility note: the LCM render is deterministic (seed taken from each target
> filename), but the Gemini sampling is stochastic and server-side, so exact prompts
> vary between runs. The reported headline run took ≈ 5.7 h on a single GTX 1650 and
> issued 126 Gemini calls.

### Custom run
The pipeline is a small library; you can script a different configuration, e.g.:

```python
import tp2

critic   = tp2.GeminiCritic()                                  # uses GEMINI_API_KEY
approach = tp2.refine_approach(critic=critic, beam=3, rounds=2,
                               n_proposals=2, verbose=True)
targets  = [tp2.TARGET_DIR / "7836.png"]                       # subset of targets
run_dir, rows, summary = tp2.run_pipeline(
    approach, targets=targets, k=3, optimiser_seeds=(0, 1, 2),
)
print(run_dir)
```

---

## 5. Outputs

Each run creates a timestamped folder `outputs/<timestamp>_<id>/`:

```
outputs/<timestamp>_final_3seeds/
├── metrics.csv / metrics.json     # top-3 per target (ranked) with all metrics
├── summary.csv / summary.json     # mean ± std across targets (best-per-target & all)
└── <target_stem>/                 # one folder per target
    ├── target.png                 # copy of the target image
    ├── rank_1.png  rank_2.png  rank_3.png   # the 3 ranked LCM reconstructions
    └── metrics.csv                # per-candidate metric table for this target
```

The three mandatory metrics are computed between each target and each render:
**CLIP image–image similarity** (`openai/clip-vit-large-patch14`, cosine of
ℓ2-normalised embeddings), **LPIPS** (AlexNet backbone), and **pixel RMSE** (RGB
normalised to `[0,1]`). The composite ranking score is
`0.4·CLIP + 0.3·(1−LPIPS) + 0.3·(1−RMSE)`.

---

## 6. Report

The LNCS report and the figures used in it are in [`report/`](report/)
(`report/main.tex`, images under `report/images/`). It can be compiled on Overleaf
with the Springer LNCS template.

---

## 7. Acknowledged external components

LCM checkpoint `SimianLuo/LCM_Dreamshaper_v7`; Google `gemini-3.1-flash-lite`
(vision–language seeder/critic, public API); `openai/clip-vit-large-patch14` and the
LPIPS/AlexNet metric; the Hugging Face `diffusers`/`transformers`, PyTorch and
`lpips` libraries. The generative-AI coding assistant *Claude Code* (Anthropic) was
used during development to help write and refactor the supporting Python code; all
design decisions, experiments and analysis are the authors' own.
