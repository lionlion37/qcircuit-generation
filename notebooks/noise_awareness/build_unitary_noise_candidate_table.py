#!/usr/bin/env python3
from __future__ import annotations

import argparse

from notebooks.noise_awareness.unitary_noise_analysis_helper import (
    DEFAULT_P_GRID,
    build_model_specs,
    collect_candidate_tables,
    save_analysis_bundle,
    summarize_candidate_tables,
)
from notebooks.noise_awareness.unitary_noise_common import maybe_dataframe, parse_float_list


def main():
    parser = argparse.ArgumentParser(description="Build a reusable candidate table for noise-aware unitary relabeling.")
    parser.add_argument("--dataset", required=True, help="Path to an existing saved unitary dataset.")
    parser.add_argument("--model-dir", action="append", default=[], help="Existing trained model directory. Can be passed multiple times.")
    parser.add_argument("--target-limit", type=int, default=32, help="Number of target unitaries to analyze.")
    parser.add_argument("--samples-per-target", type=int, default=32, help="Number of candidates to sample per target per model.")
    parser.add_argument("--noise-ps", default="0.0,0.01,0.03,0.1", help="Comma-separated global noise levels.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Guidance scale for model sampling.")
    parser.add_argument("--auto-batch-size", type=int, default=64, help="Sampling batch size.")
    parser.add_argument("--clean-infidelity-threshold", type=float, default=1e-6, help="Candidates above this threshold are not eligible for relabeling.")
    parser.add_argument("--noise-realizations", type=int, default=8, help="Number of quditkit noise realizations per candidate and p.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device for loading pipelines and datasets.")
    parser.add_argument("--output-dir", required=True, help="Output directory for candidate tables and payload.")
    args = parser.parse_args()

    model_specs = build_model_specs(args.model_dir)
    analysis = collect_candidate_tables(
        args.dataset,
        model_specs=model_specs,
        target_limit=args.target_limit,
        samples_per_target=args.samples_per_target,
        p_grid=parse_float_list(args.noise_ps) if args.noise_ps else DEFAULT_P_GRID,
        guidance_scale=args.guidance_scale,
        auto_batch_size=args.auto_batch_size,
        clean_infidelity_threshold=args.clean_infidelity_threshold,
        noise_realizations=args.noise_realizations,
        seed=args.seed,
        device=args.device,
    )
    output_dir = save_analysis_bundle(analysis, output_dir=args.output_dir)
    summary = summarize_candidate_tables(analysis["candidate_rows"], analysis["score_rows"])

    print("Saved candidate analysis to:", output_dir)
    print(maybe_dataframe(summary["by_model"]))
    print(maybe_dataframe(summary["by_noise_p"]))


if __name__ == "__main__":
    main()
