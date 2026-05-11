# Outputs Directory

Protocol outputs that are not reusable datasets or model checkpoints.

Current layout:

- `final_results/` - validated reproducible results only.
- `temporary/` - short-lived generated files used during local runs.
- `debug/` - disposable debug plots, traces, and diagnostics.

Rules:

- `final_results/` should contain only curated, reproducible outputs.
- `temporary/` and `debug/` are disposable.
- Do not keep old scatter/debug plots in `final_results/`.
- Use deterministic names such as `filtering_topk40`, `encoder_validation_results`, or `calibration_isotonic`.

