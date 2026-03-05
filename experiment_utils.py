"""Utility condivise per riproducibilita' e tracciamento delle run."""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path | str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _git_commit(cwd: Path | str = ".") -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def init_run(
    script_path: str,
    seed: int,
    config: dict[str, Any],
    legacy_artifacts: bool = False,
    deterministic: bool = True,
) -> tuple[Path, Path, Path | None, dict[str, Any]]:
    set_global_seed(seed, deterministic=deterministic)

    script_name = Path(script_path).stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path("outputs") / f"{ts}_{script_name}"
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=False)

    legacy_plots_dir = Path("plots") if legacy_artifacts else None
    if legacy_plots_dir is not None:
        legacy_plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_root.name,
        "script": script_name,
        "seed": seed,
        "deterministic": deterministic,
        "utc_started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "config": config,
    }
    save_json(run_root / "metadata.json", metadata)
    return run_root, plots_dir, legacy_plots_dir, metadata


def save_current_figure(
    plots_dir: Path,
    filename: str,
    legacy_plots_dir: Path | None = None,
    dpi: int = 150,
) -> Path:
    import matplotlib.pyplot as plt

    out_path = plots_dir / filename
    plt.savefig(out_path, dpi=dpi)
    if legacy_plots_dir is not None:
        plt.savefig(legacy_plots_dir / filename, dpi=dpi)
    return out_path


def save_model_state(
    model: torch.nn.Module,
    run_root: Path,
    legacy_artifacts: bool = False,
    filename: str = "best_model.pt",
) -> Path:
    out_path = run_root / filename
    torch.save(model.state_dict(), out_path)
    if legacy_artifacts:
        torch.save(model.state_dict(), filename)
    return out_path


def save_results(run_root: Path, results: dict[str, Any]) -> Path:
    out_path = run_root / "results.json"
    save_json(out_path, results)
    return out_path
