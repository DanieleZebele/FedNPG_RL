#!/usr/bin/env python
"""
Export run results to pgfplots .dat tables for the report.

One file per figure. Each writes, per x value, the mean across seeds plus the
min and max, so the plot can draw a line with a band showing the full spread of
the five runs rather than a modelled interval. We prefer min/max over a normal
confidence interval here because five seeds is too few for the normal
approximation to mean much, and because the Swimmer N=1 distribution is bimodal
across seeds (Section 5), where a symmetric interval would be actively
misleading.

Per-iteration curves are smoothed with a centred rolling mean before
aggregation and then subsampled, for two reasons: the raw metric is a one- to
two-episode sample per point (Section 4), and pgfplots renders slowly above a
few hundred coordinates per series.

Usage:
    python report/scripts/export_figures.py [--results DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# scaling() below is retained but not called: that figure was dropped from the
# report for now and is expected back later.
ENVS = ("swimmer", "hopper", "humanoid")
AGENTS = (1, 2, 4, 8)
SMOOTH = 21        # centred rolling window, in iterations
MAX_POINTS = 200   # subsample target per series


def load_runs(results: Path) -> list[dict]:
    runs = []
    for d in sorted(results.iterdir()):
        meta = d / "metadata.json"
        csv = d / "metrics.csv"
        if not (meta.exists() and csv.exists()):
            continue
        try:
            m = json.loads(meta.read_text())
            t = pd.read_csv(csv)
        except Exception:
            continue
        if t.empty or "mean_episode_return" not in t.columns:
            continue
        # Skip runs still in flight. aggregate() truncates every seed to the
        # shortest one, so a single partial run would silently cut the whole
        # curve short rather than just being absent from it.
        if "completed_at" not in m:
            print(f"  skipping {d.name} (still running, "
                  f"{len(t)} iterations)")
            continue
        runs.append({"meta": m, "df": t, "name": d.name})
    return runs


def curve(df: pd.DataFrame, col: str = "mean_episode_return") -> pd.Series:
    return df[col].rolling(SMOOTH, center=True, min_periods=1).mean()


def aggregate(series: list[pd.Series]) -> pd.DataFrame:
    """Mean, min and max across seeds on the iterations every seed reached."""
    n = min(len(s) for s in series)
    a = np.vstack([s.to_numpy()[:n] for s in series])
    idx = np.unique(np.linspace(0, n - 1, min(MAX_POINTS, n)).astype(int))
    return pd.DataFrame({
        "iteration": idx + 1,
        "mean": np.nanmean(a, axis=0)[idx],
        "lo": np.nanmin(a, axis=0)[idx],
        "hi": np.nanmax(a, axis=0)[idx],
    })


def write(path: Path, df: pd.DataFrame, fmt: str = "%.4g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(df.columns) + "\n")
        for row in df.itertuples(index=False):
            f.write("\t".join(fmt % v if isinstance(v, float) else str(v) for v in row) + "\n")
    print(f"  wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


def convergence(runs: list[dict], out: Path) -> None:
    """One file per environment: mean/lo/hi per agent count, over iterations."""
    for env in ENVS:
        cols: dict[str, np.ndarray] = {}
        length = None
        for n in AGENTS:
            series = [curve(r["df"]) for r in runs
                      if re.fullmatch(rf"{env}-N{n}-s\d+", r["name"])]
            if len(series) < 2:
                continue
            agg = aggregate(series)
            if length is None:
                cols["iteration"] = agg["iteration"].to_numpy()
                length = len(agg)
            k = min(length, len(agg))
            for suffix in ("mean", "lo", "hi"):
                cols[f"N{n}_{suffix}"] = agg[suffix].to_numpy()[:k]
            cols["iteration"] = cols["iteration"][:k]
            length = k
        if len(cols) > 1:
            trimmed = {c: v[:length] for c, v in cols.items()}
            write(out / f"convergence_{env}.dat", pd.DataFrame(trimmed))


def communication(runs: list[dict], out: Path) -> None:
    """Return against cumulative uplink bytes, one series per agent count.

    Plotted against the federation total rather than the per-agent figure,
    because the per-agent cost is 2*d*8 bytes per iteration regardless of N --
    that invariance IS the O(d) claim, so a per-agent x-axis would place all
    four series on top of one another. The total is what differs, and pairing
    it with reward shows what the extra agents buy for the extra traffic.
    """
    for env in ENVS:
        cols: dict[str, np.ndarray] = {}
        for n in AGENTS:
            sel = [r for r in runs if re.fullmatch(rf"{env}-N{n}-s\d+", r["name"])]
            if len(sel) < 2:
                continue
            series = [curve(r["df"]) for r in sel]
            agg = aggregate(series)
            # bytes are deterministic given (N, d, iteration), so take the first
            comm = sel[0]["df"]["cumulative_comm_bytes"].to_numpy()
            idx = agg["iteration"].to_numpy() - 1
            cols[f"N{n}_bytes"] = comm[idx]
            for suffix in ("mean", "lo", "hi"):
                cols[f"N{n}_{suffix}"] = agg[suffix].to_numpy()
        if cols:
            length = min(len(v) for v in cols.values())
            write(out / f"communication_{env}.dat",
                  pd.DataFrame({c: v[:length] for c, v in cols.items()}))


def scaling(runs: list[dict], out: Path) -> None:
    """Final reward against agent count, one row per (env, N)."""
    for env in ENVS:
        rows = []
        for n in AGENTS:
            vals = [r["df"]["mean_episode_return"].tail(100).mean() for r in runs
                    if re.fullmatch(rf"{env}-N{n}-s\d+", r["name"])]
            if not vals:
                continue
            a = np.array(vals)
            rows.append({"agents": n, "mean": a.mean(), "lo": a.min(), "hi": a.max(),
                         "sd": a.std(ddof=1) if len(a) > 1 else 0.0, "seeds": len(a)})
        if rows:
            write(out / f"scaling_{env}.dat", pd.DataFrame(rows))


# Unquantized reference for the quantization figures: the N=2,4,8 columns of
# Table 1 (mean, sd over five seeds). Hardcoded rather than recomputed because
# the Hopper reference runs are not in the same results tree as the sweep.
REFERENCE = {
    "swimmer":  {2: (126.0, 1.3), 4: (127.0, 0.8), 8: (128.4, 0.8)},
    "hopper":   {2: (2140, 386), 4: (1658, 273), 8: (1605, 159)},
    "humanoid": {2: (3391, 796), 4: (5343, 121), 8: (6329, 334)},
}
QUANT_AGENTS = (2, 4, 8)
QUANT_BITS = (6, 8)


def load_quant_runs(results: Path) -> list[dict]:
    """Completed runs of the stochastic grid, named {env}-N{n}-s{seed}-q{bits}."""
    runs = []
    for d in sorted(results.iterdir()):
        mt = re.fullmatch(r"(\w+)-N(\d+)-s(\d+)-q(\d+)", d.name)
        meta, csv = d / "metadata.json", d / "metrics.csv"
        if not (mt and meta.exists() and csv.exists()):
            continue
        m = json.loads(meta.read_text())
        if "completed_at" not in m or m.get("quant_scheme") != "stochastic":
            continue
        runs.append({"env": mt[1], "agents": int(mt[2]), "bits": int(mt[4]),
                     "df": pd.read_csv(csv), "num_params": m["num_params"]})
    return runs


def quantization(results: Path, out: Path) -> None:
    """Two files per environment from the stochastic quantization grid.

    quant_comm_<env>.dat: return against cumulative uplink at N=8, one series
    per bit width, mean/min/max across seeds (the 64-bit curve is the N=8
    series of communication_<env>.dat).
    quant_tradeoff_<env>.dat: final return (mean, sd across seeds) against
    per-agent uplink per iteration, one row per bit width including 64.
    """
    runs = load_quant_runs(results)
    print(f"loaded {len(runs)} quantization runs from {results}")
    for env in ENVS:
        sel = [r for r in runs if r["env"] == env]
        if not sel:
            continue
        cols: dict[str, np.ndarray] = {}
        for b in QUANT_BITS:
            grp = [r for r in sel if r["agents"] == 8 and r["bits"] == b]
            if len(grp) < 2:
                continue
            agg = aggregate([curve(r["df"]) for r in grp])
            idx = agg["iteration"].to_numpy() - 1
            cols[f"q{b}_bytes"] = grp[0]["df"]["cumulative_comm_bytes"].to_numpy()[idx]
            for suffix in ("mean", "lo", "hi"):
                cols[f"q{b}_{suffix}"] = agg[suffix].to_numpy()
        if cols:
            length = min(len(v) for v in cols.values())
            write(out / f"quant_comm_{env}.dat",
                  pd.DataFrame({c: v[:length] for c, v in cols.items()}))

        d = sel[0]["num_params"]
        rows = []
        for b in (*QUANT_BITS, 64):
            # g_i once plus y_i once per round, R=1: 2d values per agent.
            row = {"bits": b, "bytes": 2 * d * b // 8}
            for n in QUANT_AGENTS:
                if b == 64:
                    mean, sd = REFERENCE[env][n]
                else:
                    a = np.array([r["df"]["mean_episode_return"].tail(100).mean()
                                  for r in sel if r["agents"] == n and r["bits"] == b])
                    mean, sd = a.mean(), a.std(ddof=1)
                row[f"N{n}_mean"], row[f"N{n}_sd"] = mean, sd
            rows.append(row)
        write(out / f"quant_tradeoff_{env}.dat", pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("report/figures"))
    ap.add_argument("--quant-results", type=Path, default=None,
                    help="Separate results tree holding the quantization grid "
                         "({env}-N{n}-s{seed}-q{bits} runs).")
    a = ap.parse_args()

    runs = load_runs(a.results)
    print(f"loaded {len(runs)} runs from {a.results}")
    convergence(runs, a.out)
    communication(runs, a.out)
    if a.quant_results is not None:
        quantization(a.quant_results, a.out)


if __name__ == "__main__":
    main()
