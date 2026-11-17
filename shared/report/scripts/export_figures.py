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

# scaling() and quantization() below are retained but not called: those figures
# were dropped from the report for now and are expected back later.
ENVS = ("swimmer", "hopper", "humanoid")

# Which results directories feed the report, per environment. Hopper uses its
# own configuration (reward, advantage and observation normalization, line
# search and the log-std clamp all off), run as final-hopper-N{n}-s{seed};
# Swimmer and Humanoid keep the original configuration's runs. The reasons are
# in docs/hopper_agent_scaling_findings.md and the report's appendix table.
RUN_PREFIX = {"swimmer": "swimmer", "hopper": "final-hopper", "humanoid": "humanoid"}


def is_run(name: str, env: str, n: int) -> bool:
    """True if results directory `name` is one of env's report runs at N=n."""
    return re.fullmatch(rf"{RUN_PREFIX[env]}-N{n}-s\d+", name) is not None
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
                      if is_run(r["name"], env, n)]
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
            sel = [r for r in runs if is_run(r["name"], env, n)]
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


# Figure 3 analogue: the three algorithms at N=8, per environment. Prefix
# per (environment, algorithm); Hopper's baselines were re-run under Hopper's
# configuration so that all three arms there share the same settings.
BASELINE_ALGOS = (("admm", None), ("std", "fednpg-standard"), ("ppo", "fedppo"))


def baselines(runs: list[dict], out: Path) -> None:
    """Return against iteration and against cumulative uplink, three methods, N=8."""
    for env in ENVS:
        cols: dict[str, np.ndarray] = {}
        for key, algo in BASELINE_ALGOS:
            name = (rf"{RUN_PREFIX[env]}-N8-s\d+" if algo is None
                    else rf"{RUN_PREFIX[env]}-{algo}-N8-s\d+")
            sel = [r for r in runs if re.fullmatch(name, r["name"])]
            if len(sel) < 2:
                continue
            agg = aggregate([curve(r["df"]) for r in sel])
            idx = agg["iteration"].to_numpy() - 1
            cols.setdefault("iteration", agg["iteration"].to_numpy())
            cols[f"{key}_bytes"] = sel[0]["df"]["cumulative_comm_bytes"].to_numpy()[idx]
            for suffix in ("mean", "lo", "hi"):
                cols[f"{key}_{suffix}"] = agg[suffix].to_numpy()
        if len(cols) > 1:
            length = min(len(v) for v in cols.values())
            write(out / f"baselines_{env}.dat",
                  pd.DataFrame({c: v[:length] for c, v in cols.items()}))


# Unquantized Hopper under the ORIGINAL configuration -- the configuration the
# quantized Hopper runs use. Those 20 runs were deleted after the Hopper
# configuration replaced them, so only their recorded summaries survive
# (report Section 5.3; docs/hopper_agent_scaling_findings.md section 9).
# Final reward as in Table 2: mean of last 100 iterations, mean +- sd (ddof=1).
HOPPER_ORIGINAL_64BIT = {1: (2497, 635), 2: (2140, 386), 4: (1658, 273), 8: (1605, 159)}


def _final(df: pd.DataFrame) -> float:
    return df["mean_episode_return"].tail(100).mean()


def quant_figures(runs: list[dict], qruns: list[dict], out: Path) -> None:
    """Two tables per environment for the quantization section.

    quant_agents_<env>.dat: final reward against N for 64 (unquantized), 8 and
    6 bits -- does the faster-with-more-agents claim survive quantization.
    quant_bits_<env>.dat: final reward against bit width at N=8, from the
    five-seed grid (6, 8, 64 bits) and, as a separate series, the older
    single-seed bit-width sweep.
    """
    def ms(vals):
        a = np.array(vals, dtype=float)
        return (a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, len(a))

    for env in ENVS:
        rows = []
        for n in AGENTS:
            row = {"agents": n}
            if env == "hopper":
                m, sd = HOPPER_ORIGINAL_64BIT[n]
                row.update(b64_mean=m, b64_sd=sd)
            else:
                m, sd, _ = ms([_final(r["df"]) for r in runs if is_run(r["name"], env, n)])
                row.update(b64_mean=m, b64_sd=sd)
            for b in (8, 6):
                if n == 1:   # one agent has no uplink to quantize: same point
                    row.update({f"b{b}_mean": row["b64_mean"], f"b{b}_sd": row["b64_sd"]})
                    continue
                v = [_final(r["df"]) for r in qruns
                     if re.fullmatch(rf"{env}-N{n}-s\d+-q{b}", r["name"])]
                m, sd, _ = ms(v) if v else (np.nan, np.nan, 0)
                row.update({f"b{b}_mean": m, f"b{b}_sd": sd})
            rows.append(row)
        write(out / f"quant_agents_{env}.dat", pd.DataFrame(rows))

        five = []
        for b in (6, 8, 64):
            if b == 64:
                m, sd = (HOPPER_ORIGINAL_64BIT[8] if env == "hopper" else
                         ms([_final(r["df"]) for r in runs if is_run(r["name"], env, 8)])[:2])
            else:
                m, sd, _ = ms([_final(r["df"]) for r in qruns
                               if re.fullmatch(rf"{env}-N8-s\d+-q{b}", r["name"])])
            five.append({"bits": b, "mean": m, "sd": sd})
        write(out / f"quant_bits5_{env}.dat", pd.DataFrame(five))
        single = sorted(
            ({"bits": int(r["meta"]["quant_bits"]), "reward": _final(r["df"])}
             for r in qruns
             if r["name"].startswith(f"fednpg_admm_quant_stochastic_")
             and r["meta"].get("env") == env and r["meta"].get("quant_bits")),
            key=lambda x: x["bits"])
        if single:
            write(out / f"quant_bits1_{env}.dat", pd.DataFrame(single))


def scaling(runs: list[dict], out: Path) -> None:
    """Final reward against agent count, one row per (env, N)."""
    for env in ENVS:
        rows = []
        for n in AGENTS:
            vals = [r["df"]["mean_episode_return"].tail(100).mean() for r in runs
                    if is_run(r["name"], env, n)]
            if not vals:
                continue
            a = np.array(vals)
            rows.append({"agents": n, "mean": a.mean(), "lo": a.min(), "hi": a.max(),
                         "sd": a.std(ddof=1) if len(a) > 1 else 0.0, "seeds": len(a)})
        if rows:
            write(out / f"scaling_{env}.dat", pd.DataFrame(rows))


def quantization(results: Path, out: Path) -> None:
    """Reward and uplink bytes against bit width, from the quantization sweep."""
    rows = []
    for d in sorted(results.glob("*quant*")):
        meta, csv = d / "metadata.json", d / "metrics.csv"
        if not (meta.exists() and csv.exists()):
            continue
        m = json.loads(meta.read_text())
        t = pd.read_csv(csv)
        if t.empty or m.get("quant_bits") is None:
            continue
        rows.append({"env": m["env"], "bits": int(m["quant_bits"]),
                     "reward": t["mean_episode_return"].tail(100).mean(),
                     "bytes": t["comm_bytes_this_iteration"].iloc[0]})
    if not rows:
        print("  (no quantization runs found)")
        return
    df = pd.DataFrame(rows).sort_values(["env", "bits"])
    for env, g in df.groupby("env"):
        write(out / f"quant_{env}.dat", g[["bits", "reward", "bytes"]].reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("report/figures"))
    ap.add_argument("--quant-results", type=Path, default=None,
                    help="Separate results tree holding the quantization sweep.")
    a = ap.parse_args()

    runs = load_runs(a.results)
    print(f"loaded {len(runs)} runs from {a.results}")
    convergence(runs, a.out)
    communication(runs, a.out)
    baselines(runs, a.out)
    if a.quant_results is not None:
        qruns = load_runs(a.quant_results)
        print(f"loaded {len(qruns)} quantization runs from {a.quant_results}")
        quant_figures(runs, qruns, a.out)


if __name__ == "__main__":
    main()
