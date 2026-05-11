# Experiments Directory

Protocol-stage experiment workspaces.

Each folder corresponds to one approved research protocol stage and should contain only artifacts from that stage.

Current layout:

- `exp_001_baseline_ddpm/` - baseline latent-DDPM stage.
- `exp_002_encoder_validation/` - encoder validation and encoder selection.
- `exp_003_aligned_latent_ddpm/` - aligned latent-DDPM stage.
- `exp_004_score_calibration/` - score calibration, only if needed.
- `exp_005_filtering/` - score-based filtering.
- `exp_006_tdncnn/` - TDnCNN downstream validation.

Use deterministic names for run outputs, for example:

- `encoder_validation_results`
- `filtering_topk40`
- `calibration_isotonic`

Do not use ambiguous names such as `var`, `tmp`, `new`, `final2`, or `test_new`.
