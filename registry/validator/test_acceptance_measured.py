"""The measured GLM-5.3 acceptance profile is what its writer fits from the committed summary.

`emit_acceptance_measured.py --check` is the drift gate. Its input, the summary the writer's
`--extract` derives from the archive, is committed, so the gate runs in every CI job. The other
tests check the fit by a second path: the rates against the counters, the spread fit against
a chain whose spread is known, and the held-out cells against the files the summary read.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "emit_acceptance_measured.py"
SUMMARY = json.loads((REPO / "validator" / "testdata" / "glm53-h200-acceptance" / "summary.json").read_text())
SET = yaml.safe_load((REPO / "acceptance" / "glm53-h200-agentic.yaml").read_text())

sys.path.insert(0, str(REPO / "scripts"))
import emit_acceptance_measured as emit  # noqa: E402


def test_the_committed_set_is_what_the_writer_generates():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True,
                       cwd=str(REPO), timeout=300, env={"PATH": ""})
    assert r.returncode == 0, (
        "acceptance/glm53-h200-agentic.yaml is not what emit_acceptance_measured.py renders "
        f"from the committed summary; re-run it.\n{r.stderr}")


def test_the_rates_are_the_pooled_counters():
    [p] = SET["profiles"]
    drafts = sum(c["drafts"] for c in SUMMARY["counters"].values())
    for i, rate in enumerate(p["per_position_rates"]):
        accepted = sum(c["accepted_per_position"][i] for c in SUMMARY["counters"].values())
        assert abs(rate - accepted / drafts) < 5e-5, (i, rate, accepted / drafts)
    assert abs(p["acceptance_length"] - (1 + sum(p["per_position_rates"]))) < 1e-9


def test_no_held_out_request_record_was_read():
    held = {emit.CELL_DIR[c] for c in SUMMARY["held_out_cells"]}
    for path in SUMMARY["files"]:
        cell = path.split("/")[1]
        if cell in held:
            # Only a scrape that closes a select cell may come from a held-out cell's directory.
            assert path.endswith("/metrics-initial.prom"), path
            assert any(path.endswith(end) for _, end in emit.COUNTER_CELLS.values()), path
    assert not set(SUMMARY["held_out_cells"]) & {int(c) for c in SUMMARY["spread"]}
    assert not set(SUMMARY["held_out_cells"]) & {int(c) for c in SUMMARY["counters"]}


def _simulate(rates: list[float], sd: float, steps: list[int], merge: float, seed: int) -> dict:
    """A summary of requests drawn from the chain the profile describes, with a known spread, and
    a load tool that merges a step into the emission before with probability merge."""
    rng = random.Random(seed)
    z = emit.chain_nodes()
    base = emit.chain_base(rates, sd, z)
    sums = {"requests": 0, "sum_mean": 0.0, "sum_mean_squared": 0.0, "sum_inverse_emissions": 0.0}
    for n in steps:
        u = rng.gauss(0, 1) * sd
        q = [1 / (1 + math.exp(-(b + u))) for b in base]
        sizes: list[int] = []
        for _ in range(n):
            k = 1
            for qi in q:
                if rng.random() >= qi:
                    break
                k += 1
            if sizes and rng.random() < merge:
                sizes[-1] += k
            else:
                sizes.append(k)
        m = sum(sizes) / len(sizes)
        sums["requests"] += 1
        sums["sum_mean"] += m
        sums["sum_mean_squared"] += m * m
        sums["sum_inverse_emissions"] += 1 / len(sizes)
    return sums


@pytest.mark.parametrize("sd", [0.25, 0.6])
def test_the_spread_fit_recovers_a_known_spread(sd):
    rates = [0.84, 0.70, 0.59]
    rng = random.Random(3)
    steps = [rng.randint(60, 300) for _ in range(3000)]
    summary = {"spread": {"x": _simulate(rates, sd, steps, merge=0.025, seed=5)}}
    got, obs = emit.fit_spread(summary, rates)
    assert abs(got - sd) < 0.06, (got, sd)
    assert abs(obs["merged"] - 0.025) < 0.01, obs


def test_the_spread_fit_refuses_a_spread_it_cannot_explain():
    rates = [0.84, 0.70, 0.59]
    rng = random.Random(3)
    steps = [rng.randint(60, 300) for _ in range(2000)]
    # Every request has exactly the chain's mean: no more spread than a finite count of steps
    # gives, so there is no random effect to fit.
    flat = _simulate(rates, 0.0, steps, merge=0.0, seed=5)
    flat["sum_mean_squared"] = flat["sum_mean"] ** 2 / flat["requests"]
    with pytest.raises(SystemExit, match="no spread to fit"):
        emit.fit_spread({"spread": {"x": flat}}, rates)
