#!/usr/bin/env python3
"""Wrapper to run weak-to-strong experiments with controlled composition error rates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.addition_pipeline import main as w2s_main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch weak_to_strong_addition_experiment_v2.py with a specified percentage of boundary-carry "
            "composition errors retained in the pseudo labels."
        )
    )
    parser.add_argument(
        "--composition-error-percent",
        type=float,
        default=0.0,
        help="Percentage (0-100) of boundary-carry pseudo labels to retain when stitching with carry filtering.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to weak_to_strong_addition_experiment_v2.py (use '--' to separate).",
    )
    return parser.parse_args(argv)


def build_forward_args(args: argparse.Namespace) -> List[str]:
    composition_error_percent = args.composition_error_percent
    if composition_error_percent < 0.0 or composition_error_percent > 100.0:
        raise ValueError("composition-error-percent must be between 0 and 100.")

    forwarded = list(args.extra_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forwarded.append(f"--composition-error-percent={composition_error_percent}")
    has_composed_strategy = False
    for idx, token in enumerate(forwarded):
        if token == "--composed-strategy":
            has_composed_strategy = True
            break
        if token.startswith("--composed-strategy="):
            has_composed_strategy = True
            break
    if not has_composed_strategy:
        forwarded.append("--composed-strategy=with_carry_filtered")

    if "--dynamic-composed-digit-sampling" not in forwarded:
        forwarded.append("--dynamic-composed-digit-sampling")
    required_skips = [
        "--skip-strong-full",
        "--skip-strong-w2s",
        "--skip-strong-w2s-pseudo-direct",
        "--skip-weak-w2s",
        "--skip-weak-w2s-pseudo",
    ]
    for flag in required_skips:
        if flag not in forwarded:
            forwarded.append(flag)
    return forwarded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    forwarded_args = build_forward_args(args)
    w2s_main(forwarded_args)


if __name__ == "__main__":
    main()
