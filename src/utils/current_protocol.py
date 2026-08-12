from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


FILTER_FRACTION = 0.10
FILTER_VARIANTS = ("full", "topk_10", "quantile_10", "random_10")
LADDER_PSNR_DB = (10, 15, 20, 25, 30, 35, 40)
TOPK_DIRECTION = "minimum_score"
RANDOM_WITHOUT_REPLACEMENT = True
EARLY_STOPPING_DEFAULTS = {
    "enabled": True,
    "monitor": "val_loss",
    "mode": "min",
    "patience": 10,
    "min_delta": 1.0e-4,
}


class ProtocolError(ValueError):
    """Raised when a current pipeline config violates the shared protocol."""


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _close(value: Any, expected: float) -> bool:
    return math.isclose(_as_float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)


def _assert_fraction(name: str, value: Any) -> None:
    if not _close(value, FILTER_FRACTION):
        raise ProtocolError(f"{name} must be {FILTER_FRACTION:.2f}, got {value!r}")


def _assert_ladder(name: str, value: Iterable[Any] | None) -> None:
    got = tuple(int(x) for x in (value or ()))
    if got != LADDER_PSNR_DB:
        raise ProtocolError(f"{name} must be {list(LADDER_PSNR_DB)}, got {list(got)}")


def _validate_early_stopping(prefix: str, cfg: dict[str, Any]) -> None:
    if not cfg:
        raise ProtocolError(f"{prefix}.early_stopping is missing")
    if bool(cfg.get("enabled", False)) is not True:
        raise ProtocolError(f"{prefix}.early_stopping.enabled must be true")
    if str(cfg.get("monitor", "")) != EARLY_STOPPING_DEFAULTS["monitor"]:
        raise ProtocolError(f"{prefix}.early_stopping.monitor must be val_loss")
    if str(cfg.get("mode", "")) != EARLY_STOPPING_DEFAULTS["mode"]:
        raise ProtocolError(f"{prefix}.early_stopping.mode must be min")
    if int(cfg.get("patience", 0)) <= 0:
        raise ProtocolError(f"{prefix}.early_stopping.patience must be positive")
    if float(cfg.get("min_delta", -1.0)) < 0.0:
        raise ProtocolError(f"{prefix}.early_stopping.min_delta must be non-negative")


def validate_current_protocol(domain: str, cfg: dict[str, Any], *, downstream_models: Iterable[str]) -> None:
    """Validate scientific invariants shared by MNIST, CIFAR-10, and ImageNet-100."""
    _assert_ladder(f"{domain}.score_validation.psnr_ladder", deep_get(cfg, "score_validation.psnr_ladder"))

    filtering = cfg.get("filtering", {})
    if "ratio" in filtering:
        _assert_fraction(f"{domain}.filtering.ratio", filtering.get("ratio"))
    for key in ("topk_ratio", "quantile_ratio", "random_ratio"):
        if key in filtering:
            _assert_fraction(f"{domain}.filtering.{key}", filtering.get(key))
    if "topk_direction" in filtering and str(filtering["topk_direction"]) != TOPK_DIRECTION:
        raise ProtocolError(f"{domain}.filtering.topk_direction must be {TOPK_DIRECTION}")
    if bool(filtering.get("include_topk5", False)):
        raise ProtocolError(f"{domain}.filtering.include_topk5 must be false in current protocol")

    methods = deep_get(cfg, "protocol.filtering.methods", None)
    if methods is not None:
        required = {"full", "topk", "quantile_spread", "random"}
        if set(str(x) for x in methods) != required:
            raise ProtocolError(f"{domain}.protocol.filtering.methods must be {sorted(required)}, got {methods!r}")

    for model_name in downstream_models:
        if model_name == "tdncnn":
            es = deep_get(cfg, "downstream.tdncnn.early_stopping")
            if es is None:
                es = deep_get(cfg, "downstream.early_stopping")
            _validate_early_stopping(f"{domain}.downstream.tdncnn", es)
        elif model_name == "drunet":
            es = deep_get(cfg, "downstream.drunet.early_stopping")
            if es is None:
                es = deep_get(cfg, "drunet.early_stopping")
            _validate_early_stopping(f"{domain}.downstream.drunet", es)


def protocol_snapshot(domain: str, cfg: dict[str, Any], *, downstream_models: Iterable[str]) -> dict[str, Any]:
    return {
        "domain": domain,
        "corruption": {
            "type": deep_get(cfg, "protocol.corruption.type", deep_get(cfg, "corruption.type", "gaussian")),
            "mode": deep_get(cfg, "protocol.corruption.mode", deep_get(cfg, "corruption.mode", "per_image")),
            "target_psnr_db": [
                float(deep_get(cfg, "corruption.target_psnr_min_db", deep_get(cfg, "corpus.target_psnr_min_db", 10.0))),
                float(deep_get(cfg, "corruption.target_psnr_max_db", deep_get(cfg, "corpus.target_psnr_max_db", 40.0))),
            ],
        },
        "score_validation": {
            "psnr_ladder_db": list(LADDER_PSNR_DB),
            "coarse_mc_k": deep_get(cfg, "score_validation.coarse_mc_k", deep_get(cfg, "score_validation.mc.coarse_k")),
            "final_mc_k": deep_get(cfg, "score_validation.final_mc_k", deep_get(cfg, "score_validation.mc.final_k")),
            "pass_required_before_filtering": True,
        },
        "filtering": {
            "full": "100%",
            "topk": "10%",
            "topk_direction": TOPK_DIRECTION,
            "quantile_spread": "10%",
            "random": "10%",
            "random_without_replacement": RANDOM_WITHOUT_REPLACEMENT,
        },
        "downstream": {
            "models": list(downstream_models),
            "early_stopping": EARLY_STOPPING_DEFAULTS,
            "evaluation_checkpoint": "best.pt",
        },
    }


def print_protocol_summary(domain: str, cfg: dict[str, Any], *, run_id: str, device: str, downstream_models: Iterable[str]) -> None:
    payload = protocol_snapshot(domain, cfg, downstream_models=downstream_models)
    payload.update(
        {
            "run_id": run_id,
            "device": device,
            "seed": {
                "corpus": deep_get(cfg, "seeds.corpus_seed"),
                "subset": deep_get(cfg, "seeds.subset_seed", deep_get(cfg, "filtering.random_seed")),
                "model": deep_get(cfg, "seeds.model_seed"),
            },
        }
    )
    print("CURRENT PROTOCOL")
    print(json.dumps(payload, indent=2, sort_keys=True))


def stage_start(domain: str, stage: str, *, run_id: str | None = None) -> float:
    print("=" * 60)
    print(f"[{domain}] STAGE: {stage}")
    if run_id:
        print(f"Run: {run_id}")
    print("=" * 60)
    print(f"[START] {stage}")
    return time.time()


def stage_done(stage: str, start: float) -> None:
    print(f"[DONE] {stage} | elapsed={time.time() - start:.1f}s")


def stage_skip(stage: str, reason: str) -> None:
    print(f"[SKIP] {stage} | {reason}")


def stage_fail(stage: str, reason: str) -> None:
    print(f"[FAIL] {stage} | reason={reason}")


def should_log_progress(current: int, total: int | None, interval: int | None) -> bool:
    if current <= 1:
        return True
    if total is not None and current >= total:
        return True
    if interval is None or interval <= 0:
        return False
    return current % interval == 0


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def training_progress(
    domain: str,
    stage: str,
    *,
    unit: str,
    current: int,
    total: int | None = None,
    train_loss: Any = None,
    val_loss: Any = None,
    best_val_loss: Any = None,
    best_step: int | None = None,
    patience_counter: int | None = None,
    patience: int | None = None,
    lr: Any = None,
    start_time: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    elapsed = None if start_time is None else time.time() - start_time
    eta = None
    if elapsed is not None and total and current > 0:
        eta = elapsed * max(int(total) - int(current), 0) / max(int(current), 1)
    total_text = "?" if total is None else str(int(total))
    parts = [
        f"[{domain}] {stage}",
        f"{unit}={int(current)}/{total_text}",
    ]
    if train_loss is not None:
        parts.append(f"train_loss={_fmt_metric(train_loss)}")
    if val_loss is not None:
        parts.append(f"val_loss={_fmt_metric(val_loss)}")
    if best_val_loss is not None:
        parts.append(f"best_val_loss={_fmt_metric(best_val_loss)}")
    if best_step is not None:
        parts.append(f"best_step={int(best_step)}")
    if patience_counter is not None and patience is not None:
        parts.append(f"patience={int(patience_counter)}/{int(patience)}")
    if lr is not None:
        parts.append(f"lr={_fmt_metric(lr)}")
    if elapsed is not None:
        parts.append(f"elapsed={_fmt_seconds(elapsed)}")
    if eta is not None:
        parts.append(f"eta={_fmt_seconds(eta)}")
    for key, value in (extra or {}).items():
        parts.append(f"{key}={value}")
    print(" | ".join(parts))


@contextmanager
def stage_logging(domain: str, stage: str, *, run_id: str | None = None):
    start = stage_start(domain, stage, run_id=run_id)
    try:
        yield
    except Exception as error:
        stage_fail(stage, str(error))
        raise
    else:
        stage_done(stage, start)


def write_protocol_snapshot(path: Path, domain: str, cfg: dict[str, Any], *, downstream_models: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol_snapshot(domain, cfg, downstream_models=downstream_models), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
