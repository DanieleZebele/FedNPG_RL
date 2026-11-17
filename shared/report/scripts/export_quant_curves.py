#!/usr/bin/env python
"""
Export per-(environment, N) training curves comparing quantization bit widths.

One file per (environment, N) for N in 2, 4, 8 -- nine in all -- each holding
reward against iteration for the unquantized run and the stochastic 8- and
6-bit runs. Unquantized runs exchange float32 tensors (the byte counter charges
them 64 bits per value only to follow the paper's reporting convention), so
their columns are named fp32. N=1 is left out:
one agent has no uplink to quantize.

Same method as export_figures.py, whose helpers this reuses: centred rolling
mean of mean_episode_return, then mean, min and max across the five seeds, so
the band is the full seed spread. The single-seed 5-bit Humanoid runs are not
exported, since one seed gives no band.

Every run here comes from the quantization results tree, so the three series in
a file share one configuration: Hopper's own for Hopper, the original one for
Swimmer and Humanoid.

Also writes quant_overhead.dat: the uplink of one whole training run at N=8,
summed over agents, in GB, per environment and bit width. Read from the runs'
measured cumulative_comm_bytes, which is deterministic given (N, d, bits) and
so identical across seeds. The unquantized run appears twice: b64 as metered
(the paper's 64 bits per value) and fp32, half of it, for the float32 tensors
actually exchanged.

Usage (from /mnt/shared):
    python report/scripts/export_quant_curves.py [--quant-results DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from export_figures import ENVS, aggregate, curve, load_runs, write

AGENTS = (2, 4, 8)
# Column prefix -> run-name suffix. fp32 = unquantized.
BITS = (("fp32", ""), ("b8", "-q8"), ("b6", "-q6"))


def quant_curves(qruns: list[dict], out: Path) -> None:
    for env in ENVS:
        for n in AGENTS:
            cols: dict[str, np.ndarray] = {}
            for key, suffix in BITS:
                series = [curve(r["df"]) for r in qruns
                          if re.fullmatch(rf"{env}-N{n}-s\d+{suffix}", r["name"])]
                if len(series) < 2:
                    print(f"  {env} N={n} {key}: {len(series)} runs, skipped")
                    continue
                agg = aggregate(series)
                cols.setdefault("iteration", agg["iteration"].to_numpy())
                for stat in ("mean", "lo", "hi"):
                    cols[f"{key}_{stat}"] = agg[stat].to_numpy()
            if len(cols) > 1:
                length = min(len(v) for v in cols.values())
                write(out / f"quant_{env}_N{n}.dat",
                      pd.DataFrame({c: v[:length] for c, v in cols.items()}))


OVERHEAD_N = 8
ENV_LABEL = {"swimmer": "Swimmer", "hopper": "Hopper", "humanoid": "Humanoid"}


def quant_overhead(qruns: list[dict], out: Path) -> None:
    rows = []
    for env in ENVS:
        row = {"env": ENV_LABEL[env]}
        for key, suffix in BITS:
            total = {r["df"]["cumulative_comm_bytes"].iloc[-1] for r in qruns
                     if re.fullmatch(rf"{env}-N{OVERHEAD_N}-s\d+{suffix}", r["name"])}
            if len(total) != 1:
                raise SystemExit(f"{env} N={OVERHEAD_N} {key}: expected one byte total "
                                 f"shared by every seed, got {sorted(total)}")
            row[key] = total.pop() / 1e9
        # The meter charges unquantized values 64 bits each (the paper's
        # convention); the tensors themselves are float32, half that.
        row["b64"] = row["fp32"]
        row["fp32"] = row["b64"] / 2
        rows.append({k: row[k] for k in ("env", "b64", "fp32", "b8", "b6")})
    write(out / "quant_overhead.dat", pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant-results", type=Path, default=Path("zebele/results"))
    ap.add_argument("--out", type=Path, default=Path("report/figures/quant"))
    a = ap.parse_args()

    qruns = load_runs(a.quant_results)
    print(f"loaded {len(qruns)} quantization runs from {a.quant_results}")
    quant_curves(qruns, a.out)
    quant_overhead(qruns, a.out)


if __name__ == "__main__":
    main()
