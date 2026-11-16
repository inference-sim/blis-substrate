#!/usr/bin/env python3
"""Generate acceptance/glm53-h200-agentic.yaml from a measured GLM-5.3 run on 8xH200.

    emit_acceptance_measured.py --extract DATA_DIR   read the archive, write the summary, emit
    emit_acceptance_measured.py                       emit from the committed summary
    emit_acceptance_measured.py --check               exit non-zero if the committed file would change

The source is the reproducibility archive attached to inference-sim/inference-sim#2094
(ARCHIVE, sha256 ARCHIVE_SHA256): GLM-5.3 served on one 8xH200 node at TP 8 with expert
parallelism and MTP at K 3, under a closed-loop eight-round agentic workload at five load
levels. Two things are read from it, and nothing is typed in here but which cells are read:

* The per-position rates, from the engine's per-position acceptance counters: each cell's
  delta of cumulative counters between the scrape that opened it and the one that opened the
  next cell on the same pod (cell.json records why: the closing scrapes were lost).
* The spread across requests, from the load tool's per-request token timestamps. Each streamed
  emission after the first is one verification step, so a request's tokens per emission is its
  acceptance length plus the noise its own step count leaves. The fitted request_logit_sd is
  the chain's per-request random effect (contracts acceptance.ScheduleChain) that reproduces the
  observed variance of that statistic across requests, net of both noises (fit_spread).

The archive is not vendored. `--extract` reads it and writes the derived numbers, with the
sha256 of every file it read, to validator/testdata/glm53-h200-acceptance/summary.json; the
profile is generated from that summary alone, so `--check` runs in every CI job.

Held-out levels. The fit reads the 2-, 8- and 16-session cells only. The 4- and 12-session
cells are held out for the accuracy comparison of inference-sim#2094 (design §12.3); this script
never opens their per-request records, and reads the 12-session cell's opening scrape only as
the 8-session cell's closing one.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "acceptance" / "glm53-h200-agentic.yaml"
SUMMARY = REPO / "validator" / "testdata" / "glm53-h200-acceptance" / "summary.json"
SET_NAME = "glm53-h200-agentic"

ISSUE = "https://github.com/inference-sim/inference-sim/issues/2094"
ARCHIVE = "glm53-h200-underpricing-data-20261109.tar.gz"
ARCHIVE_SHA256 = "2f8dadd0f6c482e47510fa16c6a416e75b5a0622e7a538280a7d41d0ce6d412e"
ARCHIVE_ROOT = "glm53-h200-underpricing-data-20261109"

# Cell directories by concurrent sessions, in the archive's naming.
CELL_DIR = {2: "A_agentic_10pct", 4: "A_agentic_25pct", 8: "A_agentic_50pct",
            12: "A_agentic_75pct", 16: "A_agentic_100pct"}
# Counter deltas: each cell's opening scrape and the next cell's opening scrape on the same pod,
# as each cell.json's cache_hit_recovery records (8 -> 12 -> 16 -> 2 in time).
COUNTER_CELLS = {8: ("A_agentic_50pct/metrics-initial.prom", "A_agentic_75pct/metrics-initial.prom"),
                 16: ("A_agentic_100pct/metrics-initial.prom", "A_agentic_10pct/metrics-initial.prom")}
# The 2-session cell's closing scrape is not in the archive, so it gives no counters.
SPREAD_CELLS = (2, 8, 16)
HELD_OUT = (4, 12)
# A request with this many emissions or fewer is left out of the spread: its mean says little.
MIN_EMISSIONS = 21

K = 3
# The model's own generation_config.json and chat template at the catalog's revision state the
# sampling and the thinking mode the run used: the load tool set neither.
MODEL_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
GENERATION_CONFIG = f"https://huggingface.co/zai-org/GLM-5.3/raw/{MODEL_REVISION}/generation_config.json"
CHAT_TEMPLATE = f"https://huggingface.co/zai-org/GLM-5.3/raw/{MODEL_REVISION}/chat_template.jinja"

PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
DRAFTS = "vllm:spec_decode_num_drafts_total"
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


# --- extraction: reads the archive -------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counters(path: Path) -> tuple[float, dict[int, float]]:
    drafts, pos = 0.0, {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        if name == DRAFTS:
            drafts += value
        elif name == PER_POS:
            i = int(re.search(r'position="(\d+)"', labels).group(1))
            pos[i] = pos.get(i, 0.0) + value
    return drafts, pos


def _emissions(path: Path) -> list[list[int]]:
    """Each successful request's tokens per streamed emission, the first (the prefill's) dropped."""
    out = []
    for r in json.loads(path.read_text()):
        if r.get("error"):
            continue
        times = r["info"]["response_metrics"]["output_token_times"]
        sizes: list[int] = []
        last = None
        for t in times:
            if sizes and t == last:
                sizes[-1] += 1
            else:
                sizes.append(1)
            last = t
        out.append(sizes[1:])
    return out


def extract(data: Path) -> dict:
    files: dict[str, str] = {}

    def read(rel: str) -> Path:
        p = data / rel
        files[f"{ARCHIVE_ROOT}/{rel}"] = _sha256(p)
        return p

    counters = {}
    for cell, (start, end) in sorted(COUNTER_CELLS.items()):
        d0, p0 = _counters(read(start))
        d1, p1 = _counters(read(end))
        if sorted(p0) != list(range(K)) or sorted(p1) != list(range(K)):
            raise SystemExit(f"cell {cell}: the scrapes state positions {sorted(p1)}, not 0..{K - 1}")
        counters[str(cell)] = {"drafts": round(d1 - d0), "accepted_per_position":
                               [round(p1[i] - p0[i]) for i in range(K)]}
    spread = {}
    for cell in SPREAD_CELLS:
        rel = f"{CELL_DIR[cell]}/inference-perf/per_request_lifecycle_metrics.json"
        means, inv = [], []
        for sizes in _emissions(read(rel)):
            if len(sizes) + 1 < MIN_EMISSIONS:
                continue
            means.append(sum(sizes) / len(sizes))
            inv.append(1 / len(sizes))
        spread[str(cell)] = {"requests": len(means), "sum_mean": math.fsum(means),
                             "sum_mean_squared": math.fsum(x * x for x in means),
                             "sum_inverse_emissions": math.fsum(inv)}
    return {"source": ISSUE, "archive": ARCHIVE, "archive_sha256": ARCHIVE_SHA256,
            "held_out_cells": list(HELD_OUT), "min_emissions": MIN_EMISSIONS,
            "counters": counters, "spread": spread, "files": dict(sorted(files.items()))}


# --- the fit: reads the summary only ----------------------------------------------------------

def chain_nodes(n: int = 64) -> list[float]:
    """The standard-normal points contracts acceptance.ChainNodes averages over."""
    nd = statistics.NormalDist()
    return [nd.inv_cdf((j + 0.5) / n) for j in range(n)]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def chain_base(rates: list[float], sd: float, z: list[float]) -> list[float]:
    """The base logits contracts acceptance.NewChain solves: the mean over z of the probability
    that the first i drafts are accepted is rates[i-1]. Rates here are strictly inside (0, 1) and
    strictly falling, so no position is certain."""
    reach = [1.0] * len(z)
    base = []
    for p in rates:
        lo, hi = -60.0, 60.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if math.fsum(r * _sigmoid(mid + sd * x) for r, x in zip(reach, z)) / len(z) < p:
                lo = mid
            else:
                hi = mid
        b = (lo + hi) / 2
        base.append(b)
        reach = [r * _sigmoid(b + sd * x) for r, x in zip(reach, z)]
    return base


def step_moments(rates: list[float], sd: float, z: list[float]) -> tuple[float, float]:
    """Over the random effect: the variance of a request's mean tokens per step, and the mean of
    its per-step variance."""
    base = chain_base(rates, sd, z)
    mus, vs = [], []
    for x in z:
        reach, mu, m2 = 1.0, 0.0, 0.0
        for i, b in enumerate(base):
            q = _sigmoid(b + sd * x)
            mu += (i + 1) * reach * (1 - q)
            m2 += (i + 1) ** 2 * reach * (1 - q)
            reach *= q
        mu += (len(base) + 1) * reach
        m2 += (len(base) + 1) ** 2 * reach
        mus.append(mu)
        vs.append(m2 - mu * mu)
    m = math.fsum(mus) / len(z)
    return math.fsum((u - m) ** 2 for u in mus) / len(z), math.fsum(vs) / len(z)


def fit_rates(summary: dict) -> list[float]:
    drafts = sum(c["drafts"] for c in summary["counters"].values())
    return [sum(c["accepted_per_position"][i] for c in summary["counters"].values()) / drafts
            for i in range(K)]


def fit_spread(summary: dict, rates: list[float]) -> tuple[float, dict]:
    """The request_logit_sd whose chain reproduces the observed variance across requests.

    A request's tokens per emission is s times its tokens per step, where s = (observed mean) /
    AL > 1 because the load tool merges some consecutive steps into one emission (a fraction
    f = 1 - 1/s). Its variance across requests is then

        s^2 Var_u[mu(u)] + s^2 E_u[v(u)] E[1/steps] + m^2 f/(1-f) E[1/emissions],

    the spread the random effect makes, the noise of a finite number of steps, and the noise of
    how many merges a request happened to see. The fit solves it for the random effect's sd.
    """
    cells = summary["spread"].values()
    n = sum(c["requests"] for c in cells)
    mean = sum(c["sum_mean"] for c in cells) / n
    var = sum(c["sum_mean_squared"] for c in cells) / n - mean * mean
    inv_emissions = sum(c["sum_inverse_emissions"] for c in cells) / n
    al = 1 + sum(rates)
    s = mean / al
    f = 1 - 1 / s
    inv_steps = inv_emissions / s
    merge = mean * mean * f / (1 - f) * inv_emissions
    z = chain_nodes()

    def model(sd: float) -> float:
        between, within = step_moments(rates, sd, z)
        return s * s * between + s * s * within * inv_steps + merge

    if model(0) >= var:
        raise SystemExit(f"the observed variance {var:.4g} is no more than a chain with no spread "
                         f"gives ({model(0):.4g}); there is no spread to fit")
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if model(mid) < var:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, {"requests": n, "mean": mean, "sd": math.sqrt(var), "merged": f}


def _levels(cells) -> str:
    """2, 8 and 16 as "2-, 8- and 16-"."""
    xs = [f"{c}-" for c in sorted(cells)]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def generate(summary: dict) -> str:
    rates = [round(r, 4) for r in fit_rates(summary)]
    sd, obs = fit_spread(summary, rates)
    sd = round(sd, 3)
    al = round(1 + sum(rates), 4)
    drafts = sum(c["drafts"] for c in summary["counters"].values())
    counter_cells = _levels(int(k) for k in summary["counters"])
    spread_cells = _levels(int(k) for k in summary["spread"])
    held = _levels(summary["held_out_cells"])
    caveats = [
        f"Measured on one workload: inference-perf synthetic_agentic, eight-round sessions with a "
        f"2,048-token shared system prompt, about 2,000 new input tokens and 512 output tokens per "
        f"round, ignore_eos set. Category agentic names that workload; it is not SPEED-Bench's.",
        f"Source: the reproducibility archive {summary['archive']} (sha256 "
        f"{summary['archive_sha256']}) attached to {summary['source']}. "
        f"validator/testdata/glm53-h200-acceptance/summary.json records the sha256 of every file "
        f"read from it.",
        f"Per-position rates: the engine's per-position acceptance counters over the {counter_cells}"
        f"session cells, {drafts} verification steps pooled. Spread: tokens per streamed emission "
        f"across {obs['requests']} requests of the {spread_cells}session cells (mean "
        f"{obs['mean']:.3f}, sd {obs['sd']:.3f}), net of each request's finite step count and of "
        f"the {obs['merged']:.1%} of steps the load tool merged into the emission before.",
        f"The {held}session cells are held out for the accuracy comparison of {summary['source']} "
        f"and were not read by the fit.",
        "The chain's per-request random effect is a modelling choice: one normal offset per "
        "request on the logit of every conditional acceptance probability. It reproduces the "
        "spread of per-request means, not any finer structure, such as acceptance drifting "
        "within a request.",
    ]
    profile = {
        "model": "glm-5.3",
        "architecture": "GlmMoeDsaForCausalLM",
        "speculative_method": "mtp",
        "num_spec_tokens": K,
        "thinking": "on",
        "category": "agentic",
        "acceptance_length": al,
        "per_position_rates": rates,
        "schedule": "chain",
        "request_logit_sd": sd,
        "method": "measured",
        "provenance": {
            "source": summary["source"],
            "benchmark": "inference-perf synthetic_agentic",
            "prompts": obs["requests"],
            "hardware": "h200",
            "date": "2026-11-06",
            "engine_image": "vLLM v0.32.0 release image",
            "tp": 8,
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "settings": {
                "sampling_source": GENERATION_CONFIG,
                "thinking_source": CHAT_TEMPLATE,
                "kv_cache_dtype": "fp8",
                "enable_expert_parallel": "true",
            },
        },
    }
    doc = {"kind": "AcceptanceProfileSet", "name": SET_NAME, "caveats": caveats,
           "profiles": [profile]}
    header = (
        "# Speculative acceptance of GLM-5.3 with MTP at K 3, measured on 8xH200.\n"
        "#\n"
        "# GENERATED by scripts/emit_acceptance_measured.py from\n"
        "# validator/testdata/glm53-h200-acceptance/summary.json.\n"
        "# Re-run it and commit the diff rather than editing values here;\n"
        "# `emit_acceptance_measured.py --check` is the drift gate.\n")
    return header + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, width=88,
                              allow_unicode=True, default_flow_style=False)


def main(argv: list[str]) -> int:
    args = argv[1:]
    if len(args) == 2 and args[0] == "--extract":
        summary = extract(Path(args[1]))
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        args = []
    summary = json.loads(SUMMARY.read_text())
    text = generate(summary)
    if args == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text() != text:
            print(f"{TARGET} differs from what emit_acceptance_measured.py generates",
                  file=sys.stderr)
            return 1
        return 0
    if args:
        print(__doc__, file=sys.stderr)
        return 2
    TARGET.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
