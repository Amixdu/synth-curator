# Build Plan — `synth-curator` (Goal #2: PyPI Library)

A pip-installable Python library that packages the multi-axiom curation framework
(Part 2 of the thesis) as a reusable research artifact. This is a **productization** of
existing, validated research code — not new research. The algorithm is frozen at **v15**
(group-scored, per-class utilities, single-draw by default); the work here is extraction,
decoupling, a clean API, shipped weights, packaging, and docs.

Source of truth for the algorithm:
- `framework/versions/v15_framework_group_scored_per_class.py` (`run_for_arch`)
- `framework/framework_utils/utils.py` (shared math: robust bounds/scoring, prerequisites)
- `metric_pipeline/` (the 13-metric portfolio + torch distributional metrics)

---

## 1. Scope & positioning

The library exposes the framework's curation pipeline in the **two modes** from Goal #2:

- **Fast mode (MVP, ships first):** User brings a synthetic pool **with metrics already
  computed** (a DataFrame/CSV of instance metrics + precomputed feature tensors for the
  distributional metrics). The library applies **shipped ISIC-2018-derived utility
  weights**, runs the stochastic sieve + distributional selection, and returns a curated
  manifest. **No training, no GPU required** for the instance path; distributional
  selection needs the precomputed feature `.pt` files (CPU is fine).

- **Full mode (Phase 2):** User brings a synthetic pool + a real reference set + their own
  classifier training loop. The library runs the **metric tournament** to derive
  **domain-specific** utility weights, then curates. Requires the user's own GPU compute.
  This is a thin orchestration layer over Fast mode + a tournament runner + a training
  callback interface.

**Non-goals (v1):** shipping generative models, shipping trained classifiers (that's the
detection app), a GUI, or claiming the weights generalize beyond dermoscopy. Document the
weights as domain-specific (per REDO-PLAN2's Release section).

---

## 2. Why the current code can't just be `pip install`-ed

The research code is HPC-shaped and must be decoupled. Concretely:

| Current pattern | Problem | Fix |
|---|---|---|
| `sys.path.append(project_root)` at top of every module | Not importable as a package | Real package with relative imports |
| One giant nested YAML (`cfg['data']['synthetic_pool']['extracted_features']`, `cfg['analysis']['metric_pipeline']['fid'][...]`) drives everything | Unusable as an API; couples to the thesis dir layout | Typed config objects + explicit function args; YAML optional |
| `sys.exit(1)` on every error (`ensure_prerequisites`, `calculate_per_class_utilities`, …) | A library must not kill the host process | Raise typed exceptions |
| `print()` for all logging | No control for library consumers | `logging` module, namespaced logger |
| `ensure_instance_scores` shells out via `subprocess.run` to `run_metric_pipeline.py` with a temp YAML | Fragile, HPC-only, non-portable | In-process metric registry (Full mode) |
| `utils/standardize_images.py` hardcodes `resolution = 512  # Force 512 to match EDM2 Training` | Silently wrong for any generative model trained at a different resolution than EDM2 | Resolution passed in as a config field, derived from the pool's actual generation resolution |
| Hardcoded `CLASS_TO_IDX` (ISIC 7-class) | Not general | Class taxonomy passed in / inferred from data |
| Feature tensors loaded from hardcoded config paths via `torch.load` | Couples to thesis assets | User passes tensors/paths; loader abstraction |

The math itself — robust bounds/scoring, composite instance score, architecture-aware
utilities, distributional evaluation — is sound and ports almost verbatim.

---

## 3. Package layout

```
apps/python_library/
├── plan.md                     # this file
├── pyproject.toml              # build system, deps, entry points
├── README.md                   # quickstart + the two modes
├── src/synth_curator/
│   ├── __init__.py             # public API re-exports + __version__
│   ├── config.py               # dataclasses: CurationConfig, MetricSpec, TargetDistribution
│   ├── metrics/
│   │   ├── registry.py         # port of METRIC_PROPERTIES (type/optimize/pipeline_step)
│   │   ├── instance.py         # robust bounds/scoring, composite score
│   │   ├── distributional.py   # FID/KID/SWD over feature tensors (port torch_metrics)
│   │   └── features.py         # feature-tensor loader abstraction (dict[filename]->tensor)
│   ├── utility.py               # per-class utility weights from raw tournament predictions
│   ├── groups.py                # group distributional scores (per generation-group, inherited)
│   ├── safety.py                # optional similarity/memorization filter (pre-scoring)
│   ├── sampling.py              # draw_blocks: temperature-scaled per-class sampling
│   ├── selection.py             # per-class block selector (only used when block_search=True)
│   ├── curate.py               # high-level orchestration → CuratedResult
│   ├── weights/                # SHIPPED artifacts
│   │   └── isic2018/           # per-arch utility weights (JSON), shipped from the thesis's
│   │                           # own 30-seed run — not what run_tournament() itself defaults to
│   ├── tournament.py           # Full mode: run metric tournament, needs a TrainCallback
│   ├── io.py                   # manifest read/write (port create_csv_manifest), logging
│   ├── exceptions.py           # typed errors (replace sys.exit)
│   └── cli.py                  # `synth-curator curate ...` thin CLI
└── tests/
    ├── fixtures/               # tiny synthetic pool + fake feature tensors + tournament csv
    ├── test_instance.py
    ├── test_sampling_repro.py  # seed formula reproducibility (critical)
    ├── test_utility.py
    ├── test_selection.py
    └── test_curate_fast.py     # end-to-end fast mode on fixtures
```

---

## 4. Public API (target surface)

Fast mode — the headline call:

```python
from synth_curator import curate, load_isic2018_weights

result = curate(
    pool=pool_df,                     # DataFrame: filename, target_class_label, + instance metric cols
    features={                        # precomputed distributional features (dict or paths)
        "dino":      {"real": real_dino.pt,  "synthetic": synth_dino.pt},
        "inception": {"real": real_inc.pt,   "synthetic": synth_inc.pt},
    },
    weights=load_isic2018_weights(arch="resnet50"),  # shipped per-class weights, or from Full mode
    target_distribution=TARGET_DIST,  # class proportions
    block_size=8012,
    utility_formula="worst_anchored", # primary formula: weights ALL metrics, nothing zeroed,
                                       # fixes the instance-vs-distributional balance
    dist_utility_formula="random_anchored",  # reshapes the *already-fixed* distributional
                                       # share among distributional metrics only - can zero
                                       # an individual one, never touches the inst/dist split
    group_by=["source_checkpoint_kimg", "cfg_scale"],  # [] disables group terms
    block_search=False,               # False = v15 default (single draw, no per-block selection)
    num_blocks=1,                     # only used when block_search=True
    temperature=0.05,                 # the locked value for both resnet50 and swin_b
    safety_max_similarity_threshold=0.97,  # absolute, not a percentile; None disables it
    seed=0,
)

result.manifest         # DataFrame: source_path, target_class, safe_filename
result.utility_weights  # per-class weights actually used
result.block_log        # per-block/class U_inst / U_dist diagnostics
result.to_csv(path)     # writes the manifest
```

Full mode:

```python
from synth_curator import run_tournament, curate

weights = run_tournament(
    pool=pool_df,
    real_reference=real_df,
    train_fn=my_train_and_eval,   # user callback: (train_manifest) -> raw predictions
                                   # (unique_image_id, true_label, predicted_label)
    metrics=[...],                # which of the 13 to sweep, one single-metric arm each
    archs=["resnet50", "swin_b"],
    seed=0,                       # one seed per arm: a weighting signal, not a stats claim
    utility_formula="worst_anchored",
    dist_utility_formula="random_anchored",
)
result = curate(pool=pool_df, weights=weights["resnet50"], ...)
```

Design rules:
- Every heavy input accepted as **both** an in-memory object and a path.
- Pure functions where possible; the orchestrator wires them.
- No `sys.exit`, no `print`; raise `SynthCuratorError` subclasses, log via `logging`.

---

## 5. What ports directly vs. what gets rewritten

**Ports ~verbatim (just decouple I/O + errors):**
- `calculate_robust_bounds`, `calculate_robust_score`, `calculate_per_class_score`
- `calculate_per_class_utilities` (take raw predictions as a DataFrame/dict, not a directory
  path; raise instead of exit)
- `compute_group_scores` (group distributional scores per generation-group, e.g. checkpoint)
- `apply_safety_filter` (optional pre-scoring similarity/memorization filter)
- `draw_blocks` — **preserve the exact seed formula**
  (`seed_offset*100_000 + block_idx*100 + CLASS_TO_IDX[cls]`). Reproducibility parity with the
  thesis runs is a hard requirement.
- `evaluate_distributional_utility_per_class` / `select_optimal_block_per_class` (only exercised
  when `block_search=True`; take feature dicts as args instead of `cfg[...]` + `torch.load`)
- `create_csv_manifest` → `io.write_manifest`

**Rewritten:**
- Config: giant YAML → `CurationConfig` dataclass (YAML becomes an optional loader).
- `ensure_instance_scores`' subprocess trigger → in-process metric computation (Full mode only; Fast mode assumes metrics present and raises a clear error if not).

**Dropped from v1:** v1-v14 (`framework/versions/`) are frozen research history, not ported —
v15 is the only shipped algorithm. Legacy `class_seive`/non-stochastic `segment_blocks` variants
stay out entirely.

---

## 6. Shipped artifacts

- `weights/isic2018/{resnet50,swin_b}.json` — the per-class utility weights derived from the
  thesis's own **30-seed metric tournament's raw predictions** (the historical, statistically-
  validated run) via `calculate_per_class_utilities`, using the locked combination
  (`formula="worst_anchored"`, `dist_formula="random_anchored"`, T=0.05,
  `group_by=[source_checkpoint_kimg, cfg_scale]`, `safety_max_similarity_threshold=0.97`) from
  `final_rerun/configs/.../1_curate/3_framework_v15.yml`. This is a
  fixed, already-computed artifact being shipped, not a statement about the library's own
  `run_tournament()` default (which runs 1 seed per arm — see Section 4). **These weights are
  already available** — Step 1 (metric tournament) is done, so M1–M5 are unblocked.
- `weights/isic2018/target_distribution.json` — the 7-class proportions.
- Datasheet in README: weights are ISIC-2018/dermoscopy-specific, no fairness validation
  performed (annotator measured worse than chance — cited as future work).
- **Ship only configs validated by the Final Experiment** (REDO-PLAN2 Release note). Nothing
  goes in `weights/` until Step 3 blesses it.

---

## 7. Milestones

- **M0 — Skeleton:** `pyproject.toml`, package scaffold, CI (lint + pytest), `__version__`. No logic.
- **M1 — Instance core:** port robust scoring + per-class composite + per-class utility, with unit tests on fixtures. Pure, no torch-GPU.
- **M2 — Distributional core:** port FID/KID/SWD over feature dicts (group scores + per-class block evaluation); hoist real-stats out of loop; tests with tiny fake tensors.
- **M3 — Sampling + selection:** port `draw_blocks` + per-class block selector; **golden-value reproducibility test** asserting identical output to the thesis `run_for_arch` (v15) on a fixed fixture + seed.
- **M4 — Fast-mode `curate()` end-to-end** on fixtures + a real ISIC slice; CLI.
- **M5 — Ship ISIC weights** (gated on 30-seed tournament results) + README quickstart + datasheet.
- **M6 — Full mode:** `run_tournament` + `TrainCallback` interface; example notebook wiring a toy classifier.
- **M7 — Release:** TestPyPI → PyPI, tagged version, DOI (Zenodo) for the paper citation.

Fast mode (M0–M5) is independently shippable and citable; Full mode (M6) can trail.

---

## 8. Testing & quality

- **Reproducibility parity** is the highest-value test: same fixture + seed must reproduce
  the thesis pipeline's selected manifest byte-for-byte (guards the seed formulas).
- Unit tests for each math function against hand-computed expectations.
- Fixtures: a ~200-row fake pool, 2 fake feature tensors, a small tournament CSV — all
  committed, no large binaries.
- No network, no GPU required for the test suite (distributional tests use tiny CPU tensors).
- Ruff + mypy; GitHub Actions matrix on py3.10–3.12.

---

## 9. Open decisions (flag before M1)

1. **Package/import name.** `synth-curator` (dist) / `synth_curator` (import) per `idea.md` — confirm, check PyPI availability.
2. **`block_search` default** — v15 defaults to a single draw (`block_search=False`, no per-block distributional selection); confirm this stays the library default vs exposing `block_search=True` as an equally-supported mode rather than a fallback.
3. **Feature extraction in-package?** Fast mode assumes features precomputed. Optionally ship a `synth_curator.features.extract(model="dinov2"|"inception")` helper (adds torch/timm weight) — or keep it out and document the expected tensor format. Recommend: document format for v1, add extractor in M6.
4. **Metric portfolio in Full mode** — vendor `metric_pipeline`'s 13 calculators into the package (heavy deps: LPIPS, CLIP, BRISQUE) vs. define a `Metric` protocol and let users bring their own. Recommend the protocol + ship the light ones.

---

## 10. Dependencies (v1, Fast mode)

Core: `numpy`, `pandas`, `scipy` (v7 Spearman), `torch` (distributional metrics + tensor
load), `pyyaml` (optional config). Keep CLIP/LPIPS/timm as **optional extras**
(`pip install synth-curator[full]`) so Fast mode stays lightweight.
