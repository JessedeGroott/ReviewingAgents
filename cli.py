#!/usr/bin/env python3
"""
CLI entrypoint for Reasoning Probe & Representation Steering.

Supports three subcommands:
1. train-probe: Extracts intermediate activations and trains linear probes across layers.
2. eval: Runs automated comparative evaluation between baseline and steered generation.
3. generate: Performs user query generation with or without steering intervention.
"""

import argparse
import collections.abc
from pathlib import Path
import sys
from typing import Any, List, Optional

import torch

from reasoning_probe_steering.benchmark.dataset import (
    FactualReasoningDataset,
    get_default_data_dir,
    load_dataset,
)
from reasoning_probe_steering.benchmark.evaluator import ComparativeEvaluator
from reasoning_probe_steering.benchmark.metrics import (
    calibrate_threshold,
    compute_classification_metrics,
    compute_roc_auc,
)
from reasoning_probe_steering.hooks.collector import ActivationCollector
from reasoning_probe_steering.models.loader import load_model_and_tokenizer
from reasoning_probe_steering.probes.linear_probe import (
    ActivationProbe,
    MassMeanProbe,
    RidgeProbe,
)
from reasoning_probe_steering.steering.controller import SteeringController
from reasoning_probe_steering.steering.generator import SteeredGenerator
from reasoning_probe_steering.steering.vector_extractor import SteeringVectorExtractor


def parse_layers(layers_arg: Optional[str]) -> List[int]:
    """Parse comma-separated layer indices into integer list."""
    if not layers_arg:
        return [1]
    return [int(x.strip()) for x in str(layers_arg).split(",") if x.strip()]


def _extract_input_ids(encoded: Any) -> torch.Tensor:
    """Extract input_ids tensor cleanly from BatchEncoding, Mapping, dict, or raw tensor."""
    if hasattr(encoded, "input_ids") or isinstance(encoded, (dict, collections.abc.Mapping)):
        raw = encoded["input_ids"] if isinstance(encoded, (dict, collections.abc.Mapping)) else getattr(encoded, "input_ids")
    else:
        raw = encoded
    if isinstance(raw, list):
        return torch.tensor([raw], dtype=torch.long)
    return raw


# ==============================================================================
# Subcommand: train-probe
# ==============================================================================

def cmd_train_probe(args: argparse.Namespace) -> int:
    """Train linear probes on activations extracted from contrastive reasoning samples."""
    print(f"[*] Loading model and tokenizer: {args.model} on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    # Load training dataset
    if args.dataset:
        train_ds = FactualReasoningDataset.from_jsonl(args.dataset)
    else:
        train_ds = load_dataset(split="train")
    print(f"[*] Loaded {len(train_ds)} training samples.")

    # Load validation dataset if provided
    val_ds = None
    if args.val_dataset:
        val_ds = FactualReasoningDataset.from_jsonl(args.val_dataset)
        print(f"[*] Loaded {len(val_ds)} validation samples.")

    target_layers = parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim = getattr(model, "hidden_dim", None)
    if hidden_dim is None and hasattr(model, "config"):
        hidden_dim = getattr(model.config, "hidden_size", 64)
    if hidden_dim is None:
        hidden_dim = 64

    # Extract activations for each layer
    for layer_idx in target_layers:
        print(f"\n[*] Collecting activations for Layer {layer_idx}...")
        collector = ActivationCollector(model=model, target_layers=[layer_idx], extract_token="last")

        pos_acts = []
        neg_acts = []

        for sample in train_ds:
            # Truthful trace activation
            truth_prompt = sample.query + " " + sample.truthful_text
            encoded_truth = tokenizer(truth_prompt, return_tensors="pt")
            input_ids_truth = _extract_input_ids(encoded_truth)
            with collector:
                model(input_ids_truth.to(args.device))
            truth_act = collector.get_activations().get(layer_idx)
            if truth_act is not None:
                pos_acts.append(truth_act.squeeze(0).cpu())

            # Hallucinated trace activation
            halluc_prompt = sample.query + " " + sample.hallucinated_text
            encoded_halluc = tokenizer(halluc_prompt, return_tensors="pt")
            input_ids_halluc = _extract_input_ids(encoded_halluc)
            with collector:
                model(input_ids_halluc.to(args.device))
            halluc_act = collector.get_activations().get(layer_idx)
            if halluc_act is not None:
                neg_acts.append(halluc_act.squeeze(0).cpu())

        if not pos_acts or not neg_acts:
            print(f"[!] Warning: No activations collected for layer {layer_idx}.")
            continue

        X_train_pos = torch.stack(pos_acts)
        X_train_neg = torch.stack(neg_acts)
        X_train = torch.cat([X_train_pos, X_train_neg], dim=0)
        y_train = torch.cat([torch.ones(len(pos_acts)), torch.zeros(len(neg_acts))], dim=0)

        # Instantiate probe
        probe_type = getattr(args, "probe_type", "logistic")
        if probe_type == "mass_mean":
            probe = MassMeanProbe(input_dim=hidden_dim, device=args.device)
        elif probe_type == "ridge":
            probe = RidgeProbe(input_dim=hidden_dim, device=args.device)
        else:
            probe = ActivationProbe(input_dim=hidden_dim, device=args.device)

        print(f"[*] Training {probe.__class__.__name__} on {len(X_train)} samples...")
        probe.fit(X_train, y_train, max_epochs=args.epochs, lr=args.lr)

        # Validation evaluation
        val_auc = 0.5
        val_acc = 0.5
        optimal_th = 0.5
        if val_ds is not None and len(val_ds) > 0:
            val_pos = []
            val_neg = []
            for sample in val_ds:
                t_enc = tokenizer(sample.query + " " + sample.truthful_text, return_tensors="pt")
                t_ids = _extract_input_ids(t_enc)
                with collector:
                    model(t_ids.to(args.device))
                act_t = collector.get_activations().get(layer_idx)
                if act_t is not None:
                    val_pos.append(act_t.squeeze(0).cpu())

                h_enc = tokenizer(sample.query + " " + sample.hallucinated_text, return_tensors="pt")
                h_ids = _extract_input_ids(h_enc)
                with collector:
                    model(h_ids.to(args.device))
                act_h = collector.get_activations().get(layer_idx)
                if act_h is not None:
                    val_neg.append(act_h.squeeze(0).cpu())

            if val_pos and val_neg:
                X_val = torch.cat([torch.stack(val_pos), torch.stack(val_neg)], dim=0)
                y_val = torch.cat([torch.ones(len(val_pos)), torch.zeros(len(val_neg))], dim=0)
                probs = probe.predict_proba(X_val)
                val_auc = compute_roc_auc(y_val, probs)
                optimal_th = calibrate_threshold(y_val, probs)
                probe.optimal_threshold = optimal_th
                metrics = compute_classification_metrics(y_val, probs, threshold=optimal_th)
                val_acc = metrics["accuracy"]

        # Extract steering vector
        steering_vec = probe.get_steering_vector(normalize=True)

        # Save artifacts
        probe_path = output_dir / f"probe_layer_{layer_idx}.pt"
        vector_path = output_dir / f"vector_layer_{layer_idx}.pt"

        torch.save({
            "state_dict": probe.state_dict(),
            "input_dim": hidden_dim,
            "threshold": optimal_th,
            "layer_idx": layer_idx,
            "val_auc": val_auc,
        }, probe_path)

        torch.save(steering_vec, vector_path)

        print(f"[+] Layer {layer_idx} Probe Saved: {probe_path}")
        print(f"[+] Layer {layer_idx} Steering Vector Saved: {vector_path}")
        print(f"    Validation ROC-AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f} | Optimal Threshold: {optimal_th:.2f}")

    print("\n[+] Training complete across all requested layers.")
    return 0


# ==============================================================================
# Subcommand: eval
# ==============================================================================

def cmd_eval(args: argparse.Namespace) -> int:
    """Run automated comparative evaluation benchmark between baseline and steered generation."""
    print(f"[*] Loading model and tokenizer: {args.model} on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    # Load test dataset
    if args.test_dataset:
        test_ds = FactualReasoningDataset.from_jsonl(args.test_dataset)
    else:
        test_ds = load_dataset(split="test")
    print(f"[*] Loaded {len(test_ds)} test samples.")

    target_layers = parse_layers(args.target_layers)
    controller = SteeringController(model=model)

    # Load vector checkpoint if specified
    if args.vector_checkpoint:
        v_path = Path(args.vector_checkpoint)
        if v_path.exists():
            vec = torch.load(v_path, map_location=args.device, weights_only=True)
            for l in target_layers:
                controller.set_steering_vector(l, vec)
            print(f"[*] Loaded steering vector from {v_path} for layers {target_layers}")

    # Load probe checkpoint if specified
    probe = None
    if args.probe_checkpoint:
        p_path = Path(args.probe_checkpoint)
        if p_path.exists():
            ckpt = torch.load(p_path, map_location=args.device, weights_only=True)
            probe = ActivationProbe(input_dim=ckpt.get("input_dim", 64), device=args.device)
            probe.load_state_dict(ckpt["state_dict"])
            probe.optimal_threshold = ckpt.get("threshold", 0.5)
            # If no explicit vector checkpoint, extract vector from probe
            if not controller.steering_vectors:
                vec = probe.get_steering_vector(normalize=True)
                for l in target_layers:
                    controller.set_steering_vector(l, vec)
            print(f"[*] Loaded probe from {p_path}")

    evaluator = ComparativeEvaluator(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        controller=controller,
        probe=probe,
        device=args.device,
    )

    print(f"[*] Running comparative evaluation (alpha={args.alpha}, max_tokens={args.max_new_tokens})...")
    report = evaluator.run_evaluation(
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        target_layers=target_layers,
    )

    if args.output_json:
        report.save_json(args.output_json)
        print(f"[+] Saved JSON report to: {args.output_json}")

    if args.output_csv:
        report.save_csv(args.output_csv)
        print(f"[+] Saved CSV report to: {args.output_csv}")

    # Print summary table to console
    print("\n")
    report.print_summary_table()
    return 0


# ==============================================================================
# Subcommand: generate
# ==============================================================================

def cmd_generate(args: argparse.Namespace) -> int:
    """Generate reasoning output for a query, with or without representation steering."""
    print(f"[*] Loading model and tokenizer: {args.model} on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)

    target_layers = parse_layers(args.target_layers)
    controller = SteeringController(model=model)

    if args.vector_checkpoint:
        v_path = Path(args.vector_checkpoint)
        if v_path.exists():
            vec = torch.load(v_path, map_location=args.device, weights_only=True)
            for l in target_layers:
                controller.set_steering_vector(l, vec)

    if args.probe_checkpoint:
        p_path = Path(args.probe_checkpoint)
        if p_path.exists():
            ckpt = torch.load(p_path, map_location=args.device, weights_only=True)
            probe = ActivationProbe(input_dim=ckpt.get("input_dim", 64), device=args.device)
            probe.load_state_dict(ckpt["state_dict"])
            if not controller.steering_vectors:
                vec = probe.get_steering_vector(normalize=True)
                for l in target_layers:
                    controller.set_steering_vector(l, vec)

    generator = SteeredGenerator(
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        device=args.device,
    )

    prompt = args.prompt

    if args.compare:
        print("\n" + "=" * 70)
        print("                  COMPARATIVE GENERATION (BASELINE vs STEERED)")
        print("=" * 70)
        print(f"PROMPT: {prompt}\n")
        res = generator.compare(prompt=prompt, alpha=args.alpha, max_new_tokens=args.max_new_tokens)
        print("-" * 70)
        print("--- [BASELINE OUTPUT] ---")
        print(res.baseline.text.strip())
        print(f"Tokens: {len(res.baseline.generated_tokens)}")
        print("\n--- [STEERED OUTPUT (alpha = {:.2f})] ---".format(res.alpha))
        print(res.steered.text.strip())
        print(f"Tokens: {len(res.steered.generated_tokens)}")
        print("-" * 70)
    else:
        steer_flag = not args.no_steer
        res = generator.generate(
            prompt=prompt,
            steer=steer_flag,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
        )
        mode_str = f"STEERED (alpha={args.alpha:.2f})" if steer_flag else "BASELINE (UNSTEERED)"
        print("\n" + "=" * 70)
        print(f"GENERATION RESULT [{mode_str}]")
        print("=" * 70)
        print(f"PROMPT: {prompt}\n")
        print("--- [OUTPUT] ---")
        print(res.text.strip())
        print(f"Tokens: {len(res.generated_tokens)}")
        print("=" * 70)

    return 0


# ==============================================================================
# Main Parser
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Reasoning Probe & Representation Steering CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. train-probe
    p_train = subparsers.add_parser("train-probe", help="Train activation probes on dataset")
    p_train.add_argument("--model", type=str, default="mock", help="Model name or 'mock'")
    p_train.add_argument("--dataset", type=str, default=None, help="Path to train.jsonl")
    p_train.add_argument("--val-dataset", type=str, default=None, help="Path to val.jsonl")
    p_train.add_argument("--layers", type=str, default="1", help="Comma-separated target layers")
    p_train.add_argument("--output-dir", type=str, default="checkpoints/probes", help="Output directory")
    p_train.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    p_train.add_argument("--probe-type", type=str, default="logistic", choices=["logistic", "ridge", "mass_mean"])
    p_train.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=0.05, help="Learning rate")

    # 2. eval
    p_eval = subparsers.add_parser("eval", help="Run comparative benchmark evaluation")
    p_eval.add_argument("--model", type=str, default="mock", help="Model name or 'mock'")
    p_eval.add_argument("--test-dataset", type=str, default=None, help="Path to test.jsonl")
    p_eval.add_argument("--probe-checkpoint", type=str, default=None, help="Probe .pt checkpoint")
    p_eval.add_argument("--vector-checkpoint", type=str, default=None, help="Steering vector .pt checkpoint")
    p_eval.add_argument("--target-layers", type=str, default="1", help="Comma-separated target layers")
    p_eval.add_argument("--alpha", type=float, default=1.5, help="Steering alpha scalar")
    p_eval.add_argument("--max-new-tokens", type=int, default=16, help="Max generated tokens")
    p_eval.add_argument("--output-json", type=str, default=None, help="Output JSON report path")
    p_eval.add_argument("--output-csv", type=str, default=None, help="Output CSV report path")
    p_eval.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    # 3. generate
    p_gen = subparsers.add_parser("generate", help="Generate reasoning traces with or without steering")
    p_gen.add_argument("--model", type=str, default="mock", help="Model name or 'mock'")
    p_gen.add_argument("--prompt", type=str, required=True, help="User reasoning prompt")
    p_gen.add_argument("--probe-checkpoint", type=str, default=None, help="Probe .pt checkpoint")
    p_gen.add_argument("--vector-checkpoint", type=str, default=None, help="Steering vector .pt checkpoint")
    p_gen.add_argument("--target-layers", type=str, default="1", help="Comma-separated target layers")
    p_gen.add_argument("--alpha", type=float, default=1.5, help="Steering alpha scalar")
    p_gen.add_argument("--max-new-tokens", type=int, default=16, help="Max generated tokens")
    p_gen.add_argument("--compare", action="store_true", help="Run side-by-side comparison")
    p_gen.add_argument("--no-steer", action="store_true", help="Disable steering")
    p_gen.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-probe":
        sys.exit(cmd_train_probe(args))
    elif args.command == "eval":
        sys.exit(cmd_eval(args))
    elif args.command == "generate":
        sys.exit(cmd_generate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
