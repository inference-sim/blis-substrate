#!/usr/bin/env python3
"""Fit the encoder (vision-tower) attention primitive from AISimulate's encoder_attention sweeps.

The sweeps time a non-causal, no-KV-cache attention kernel (vLLM's ViT flash attention: FA3 on
Hopper, FA4 on Blackwell) over batch, sequence length, head count and head dimension (64, 72 or
80), in bf16. The form is the prefill form (scripts/fit_attention_prefill.py) over every
query-key pair instead of the causal half:

    latency = floor + 4 * b * n_heads * head_dim * isl^2 / (peak * eff(tokens) * work_scale)

with eff the registry's bf16 GEMM ramp. Both parameters are fitted per part by minimising the mean
absolute log ratio. Prints one line per part; the registry entries carry the values, each citing
the parquet by path and sha256.

Usage:
    python scripts/fit_attention_encoder.py [--data <dir>] [--registry <dir>] [--catalog <dir>]
        [--collection vllm/0.30.0] [sku:chip ...]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from fit_attention_prefill import DEFAULT_CATALOG, DEFAULT_REGISTRY, FLOOR_GRID, SCALE_GRID, SKUS, envelope

DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/Users/sri/Documents/Projects/.blis-m3-tools/pulls/aisim-6fdf6fb6/repo/python/aisim/src/aisim_core/systems/data",
)
DEFAULT_COLLECTION = "vllm/0.30.0"


def fit(data: Path, catalog: Path, sku: str, chip: str, env: tuple[float, float], collection: str) -> str:
    path = data / sku / "encoder_attention" / collection / "encoder_attention_perf.parquet"
    if not path.exists():
        return f"{chip:14s} no encoder-attention sweep at {collection}"
    d = pq.read_table(path).to_pydict()
    hw = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = hw["TFlopsPeak"] * 1e12
    eps, m_half = env
    pts = []
    for i in range(len(d["latency"])):
        if d["latency"][i] <= 0:
            continue
        b, isl, nq, dh = d["batch_size"][i], d["isl"][i], d["num_heads"][i], d["head_dim"][i]
        tok = b * isl
        flops = 4.0 * b * nq * dh * isl * isl
        eff = eps * tok / (tok + m_half)
        if eff <= 0:
            continue
        pts.append((flops / (peak * eff), d["latency"][i] * 1e-3))
    if not pts:
        return f"{chip:14s} no usable rows"
    best = None
    for floor in FLOOR_GRID:
        for scale in SCALE_GRID:
            err = math.exp(sum(abs(math.log((floor * 1e-6 + w / scale) / lat)) for w, lat in pts) / len(pts))
            if best is None or err < best[0]:
                best = (err, floor, scale)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return (f"{chip:14s} n={len(pts):6d} floor={best[1]:5.1f}us work_scale={best[2]:.2f} "
            f"geo-err {best[0]:.3f}x  {path.relative_to(data)} sha256 {sha}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("parts", nargs="*", metavar="sku:chip")
    args = ap.parse_args(argv[1:])
    env = envelope(Path(args.registry))
    pairs = ([(s.partition(":")[0], s.partition(":")[2] or SKUS.get(s.partition(":")[0], "")) for s in args.parts]
             if args.parts else sorted(SKUS.items()))
    print(f"# lane: {args.collection}")
    rc = 0
    for sku, chip in pairs:
        if not chip or chip not in env:
            print(f"error: {sku}: no chip or no bf16 envelope in the registry", file=sys.stderr)
            rc = 1
            continue
        print(fit(Path(args.data), Path(args.catalog), sku, chip, env[chip], args.collection))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
