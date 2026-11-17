# Shared results for the final report

Docs: `edoardo/docs/` (findings, configuration, theory, repo README) and
`zebele/docs/` (quantization report, repo README). The theory doc was
identical in both repos and is kept once, under edoardo/docs.

Copied from both working copies of FedNPG-ADMM-FedRL by
`collect_shared_results.sh` (in the edoardo repo). The originals are untouched.
Every run here completed; killed and development runs were left behind.
Each run directory holds `metadata.json` (all settings) and `metrics.csv`
(one row per iteration). Read `ep_rew_mean` or the report's metric
(`mean_episode_return`, mean of the last 100 iterations), not single points.

## report/ -- the LaTeX report

Sources, figures, the export script and the compiled `main.pdf`, copied from
the edoardo repo (where it is not under version control). Build with
`cd report && latexmk -pdf main.tex`. To regenerate the figure data from the
results in this folder (the script defaults assume the repo layout):

    cd /mnt/shared
    python report/scripts/export_figures.py --results edoardo/results --out report/figures

## edoardo/results/ -- reproduction and baselines (90 runs, 5 seeds each)

| runs | what | configuration |
|---|---|---|
| `swimmer-N{1,2,4,8}-s{0..4}` | FedNPG-ADMM, Swimmer | original |
| `humanoid-N{1,2,4,8}-s{0..4}` | FedNPG-ADMM, Humanoid | original |
| `final-hopper-N{1,2,4,8}-s{0..4}` | FedNPG-ADMM, Hopper | Hopper |
| `{swimmer,humanoid}-{fednpg-standard,fedppo}-N8-s{0..4}` | baselines | original |
| `final-hopper-{fednpg-standard,fedppo}-N8-s{0..4}` | baselines, Hopper | Hopper |

"Original" = Table 4 plus observation, reward and advantage normalization,
TRPO line search and the log-std clamp [-2, 0.5]. "Hopper" = the same with
all five of those off: reward and advantage normalization made Hopper
collapse late in training, worse with more agents. Details:
docs/hopper_agent_scaling_findings.md in the repo.

## zebele/results/ -- quantization (154 runs)

| runs | what |
|---|---|
| `{env}-N{2,4,8}-s{0..4}-q{6,8}` | stochastic quantization, 6 and 8 bits, 5 seeds |
| `{env}-N1-s{0..4}` | N=1, unquantized (one agent has no uplink to quantize) |
| `hopper-N{2,4,8}-s{0..4}` | Hopper unquantized references (copies of `final-hopper-*`) |
| `humanoid-N{2,4,8}-s0-q5` | 5 bits, single seed |
| `fednpg_admm_quant_stochastic_<bits>bit_<env>_8agents_*` | single-seed bit-width scan at N=8 (the report's threshold scan) |

Swimmer and Humanoid quantization runs use the original configuration; Hopper runs use the Hopper configuration (same as `final-hopper-*`).

## Which unquantized runs to compare against

- Swimmer, Humanoid: `edoardo/results/{env}-N{n}-s*` -- same configuration
  as the quantized runs (checked field by field; only bookkeeping fields
  differ).
- Hopper: `zebele/results/hopper-N{n}-s*` (identical to
  `edoardo/results/final-hopper-N{n}-s*`), same Hopper configuration and seeds
  as the quantized Hopper runs.
