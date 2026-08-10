# Experiment Protocol

Current corruption:

```text
target_PSNR ~ Uniform(10, 40) dB
sigma = 10^(-target_PSNR / 20)
x_corpus = clip(x_clean + sigma * N(0, I), 0, 1)
```

Current score:

```text
score(x, t) = ||epsilon_theta(x_t, t)||^2
```

Score validation scans checkpoint/timestep candidates, refines top candidates with Monte Carlo scoring, and writes a PASS/FAIL decision. Filtering and downstream stages require PASS.

Downstream training uses validation-based early stopping and evaluates `best.pt`.
