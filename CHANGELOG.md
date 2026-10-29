# Changelog

## 0.1.0 — initial release

- Full pipeline: standardize → extract features → compute metrics → curate metric sets → run
  the tournament → derive per-class weights → curate the final set (`SynthCurator.run()`).
- Two-tier training: cheap tournament ranking (`train_and_evaluate`) vs. an optional, one-time
  production training call (`train_production`).
- Pluggable scoring (`metric_beta` / `scoring`) and utility formulas (`utility_formula` /
  `dist_utility_formula`).
- Multi-GPU tournament dispatch (`model_factory` + `gpu_ids`).
- Precomputed weights (`weights_path=` / `shipped_weights=`) to skip the tournament entirely —
  ships with two ISIC-2018 presets (`isic2018_swin_b`, `isic2018_resnet50`).
- `resample()` for model-locked, data-variance-swept validation sweeps, with matching config/CLI
  support (`resample:` section).
- Config file (`config.example.yaml` / `config.example.shipped.yaml`) + `synth-curator` CLI as an
  alternative to the Python API.
- Fingerprint-based resumability on every cacheable step; `pool_workdir` for sharing steps 1-3
  across a sweep.
