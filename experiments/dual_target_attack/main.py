"""
Main experiment script for dual-target adversarial attack
"""

import argparse
import torch
import numpy as np
import random
import os
import sys
from tqdm import tqdm
import json
from datetime import datetime
from scipy.io import wavfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.speaker_model import load_speaker_model
from models.purification import load_purification_model
from attacks.dual_pgd import DualTargetPGD, SingleTargetPGD, AdaptiveWeightPGD
from attacks.diffattack_pgd import DiffAttackPGD
from attacks.fulltrace_pgd import FullTraceDiffAttackPGD
from data.dataset import get_dataloader
from evaluation.metrics import AttackMetrics


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(config, attack_type="dual"):
    """
    Run adversarial attack experiment

    Args:
        config: Configuration object
        attack_type: 'dual', 'single', or 'adaptive'
    """
    print(f"\n{'=' * 60}")
    print(f"Running {attack_type.upper()} Target Attack Experiment")
    print(f"{'=' * 60}\n")

    # Set seed
    set_seed(config.seed)

    # Load models
    print("Loading models...")
    speaker_model = load_speaker_model(
        model_path=config.speaker_model_path,
        model_type=getattr(config, "speaker_model_type", "ecapa"),
        device=config.device,
        input_sr=config.sample_rate,
    )
    eval_speaker_model = load_speaker_model(
        model_path=getattr(config, "eval_speaker_model_path", config.speaker_model_path),
        model_type=getattr(config, "eval_speaker_model_type", "ecapa"),
        device=config.device,
        input_sr=config.sample_rate,
    )
    xvector_model = load_speaker_model(
        model_path=getattr(config, "xvector_model_path", None),
        model_type=getattr(config, "xvector_model_type", "xvector"),
        device=config.device,
        input_sr=config.sample_rate,
    )

    purification_model = load_purification_model(
        model_type=config.purification_type,
        model_path=config.purification_model_path,
        device=config.device,
        reverse_timestep=config.purification_reverse_timestep,
        step_stride=config.purification_step_stride,
        use_checkpoint=config.purification_use_checkpoint,
    )

    # Load data
    print("Loading dataset...")
    dataloader = get_dataloader(config, split="test")

    # Initialize attack
    print(f"Initializing {attack_type} attack...")
    if attack_type == "dual":
        attacker = DualTargetPGD(speaker_model, purification_model, config)
    elif attack_type == "single":
        attacker = SingleTargetPGD(speaker_model, config)
    elif attack_type == "adaptive":
        attacker = AdaptiveWeightPGD(speaker_model, purification_model, config)
    elif attack_type == "diffattack":
        attacker = DiffAttackPGD(speaker_model, purification_model, config)
    elif attack_type == "fulltrace":
        attacker = FullTraceDiffAttackPGD(speaker_model, purification_model, config)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Initialize metrics
    metrics_evaluator = AttackMetrics(
        eval_speaker_model,
        purification_model,
        config.device,
        sample_rate=config.sample_rate,
        xvector_model=xvector_model,
        ecapa_sva_threshold=getattr(config, "ecapa_sva_threshold", 0.75),
        xvector_sva_threshold=getattr(config, "xvector_sva_threshold", 0.75),
        resemblyzer_sva_threshold=getattr(
            config, "resemblyzer_sva_threshold", 0.75
        ),
    )

    # Run attack on all samples
    print("Running attacks...")
    all_metrics = []
    os.makedirs(config.log_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        config.log_dir, f"{attack_type}_attack_results_{run_ts}.json"
    )

    for batch_idx, (x_clean, labels, file_paths) in enumerate(tqdm(dataloader)):
        x_clean = x_clean.to(config.device)
        labels = labels.to(config.device)

        target_embed = None
        if attack_type != "diffattack":
            # Find a target sample with a different label for each source
            target_indices = []
            for i in range(len(labels)):
                diff = (labels != labels[i]).nonzero(as_tuple=True)[0]
                if len(diff) > 0:
                    target_indices.append(diff[0].item())
                else:
                    # fallback: use roll if all labels are the same
                    target_indices.append((i + 1) % len(labels))
            target_audio = x_clean[target_indices]
            with torch.no_grad():
                target_embed = speaker_model.get_embedding(target_audio)

        # Execute attack
        if attack_type == "single":
            x_adv = attacker.attack(x_clean, target_embed)
        elif attack_type in ("diffattack", "fulltrace"):
            x_adv, trajectory = attacker.attack(x_clean, return_trajectory=True)
        else:
            x_adv, trajectory = attacker.attack(
                x_clean, target_embed, return_trajectory=True
            )

        # Compute metrics
        batch_metrics = metrics_evaluator.compute_all_metrics(
            x_clean,
            x_adv,
            target_embed,
            config.sample_rate,
            ppr_threshold=getattr(config, "ppr_threshold", 0.5),
        )
        batch_metrics["file_paths"] = list(file_paths)
        all_metrics.append(batch_metrics)

        # Save audio samples
        if getattr(config, "save_audio", True):
            audio_base = os.path.join(config.audio_output_dir, f"{attack_type}_{run_ts}")
            adv_dir = os.path.join(audio_base, "adv")
            purified_dir = os.path.join(audio_base, "purified")
            os.makedirs(adv_dir, exist_ok=True)
            os.makedirs(purified_dir, exist_ok=True)
            with torch.no_grad():
                x_purified = metrics_evaluator.purification_model(x_adv)

            def save_wav(path, tensor):
                arr = (tensor.squeeze().cpu().numpy() * 32767).astype(np.int16)
                wavfile.write(path, config.sample_rate, arr)

            for i in range(x_clean.shape[0]):
                fname = os.path.basename(file_paths[i])
                save_wav(os.path.join(adv_dir, f"adv_{fname}"), x_adv[i])
                save_wav(
                    os.path.join(purified_dir, f"purified_{fname}"),
                    x_purified[i].squeeze(0),
                )

        # Log progress and checkpoint
        if batch_idx % config.log_interval == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{ts}] Batch {batch_idx}: ASR={batch_metrics['asr']:.2%}, "
                f"PPR={batch_metrics['ppr']:.2%}, "
                f"PurifiedSim(source)={batch_metrics['purified_source_sim']:.4f}"
            )
            # Intermediate save
            with open(result_file, "w") as f:
                json.dump(
                    {"attack_type": attack_type, "per_sample_metrics": all_metrics},
                    f,
                    indent=2,
                )

    # Aggregate metrics
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
        if key != "file_paths"
    }

    metrics_evaluator.print_metrics(avg_metrics)

    # Save results
    results = {
        "attack_type": attack_type,
        "config": {
            "speaker_model_type": getattr(config, "speaker_model_type", "ecapa"),
            "eval_speaker_model_type": getattr(
                config, "eval_speaker_model_type", "ecapa"
            ),
            "xvector_model_type": getattr(config, "xvector_model_type", "xvector"),
            "ecapa_sva_threshold": getattr(config, "ecapa_sva_threshold", 0.75),
            "xvector_sva_threshold": getattr(config, "xvector_sva_threshold", 0.75),
            "resemblyzer_sva_threshold": getattr(
                config, "resemblyzer_sva_threshold", 0.75
            ),
            "epsilon": config.epsilon,
            "num_iterations": config.num_iterations,
            "alpha": config.alpha,
            "beta": config.beta,
        },
        "avg_metrics": avg_metrics,
        "per_sample_metrics": all_metrics,
        "success": bool(
            avg_metrics["asr"] > config.target_asr
            and avg_metrics["ppr"] > config.target_ppr
        ),
    }

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {result_file}")

    # Free GPU memory before next experiment
    del speaker_model, purification_model, attacker, metrics_evaluator
    del eval_speaker_model
    del xvector_model
    torch.cuda.empty_cache()

    return avg_metrics, results["success"]


def run_ablation_study(config):
    """Run ablation study on different hyperparameters"""
    print("\n" + "=" * 60)
    print("Running Ablation Study")
    print("=" * 60 + "\n")

    # Test different epsilon values
    epsilons = [0.01, 0.02, 0.05]

    # Test different weight combinations
    weight_combinations = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]

    results = []

    for eps in epsilons:
        for alpha, beta in weight_combinations:
            print(f"\nTesting epsilon={eps}, alpha={alpha}, beta={beta}")

            # Update config
            config.update(epsilon=eps, alpha=alpha, beta=beta)

            # Run experiment
            metrics, success = run_experiment(config, attack_type="dual")

            results.append(
                {
                    "epsilon": eps,
                    "alpha": alpha,
                    "beta": beta,
                    "metrics": metrics,
                    "success": success,
                }
            )

    # Save ablation results
    ablation_file = os.path.join(config.log_dir, "ablation_study.json")
    with open(ablation_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation study results saved to {ablation_file}")

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack-type",
        "--attack_type",
        dest="attack_type",
        choices=["all", "single", "dual", "adaptive", "diffattack", "fulltrace"],
        default="diffattack",
        help="Which experiment flow to run.",
    )
    parser.add_argument("--data-root", dest="data_root", help="Override dataset root.")
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, help="Override sample count."
    )
    parser.add_argument(
        "--num-iterations",
        dest="num_iterations",
        type=int,
        help="Override PGD iterations for quick smoke tests.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.data_root:
        overrides["data_root"] = args.data_root
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.num_iterations is not None:
        overrides["num_iterations"] = args.num_iterations
    if overrides:
        config.update(**overrides)

    print("\n" + "=" * 60)
    print("Dual-Target Adversarial Attack Experiment")
    print("=" * 60 + "\n")

    # Create output directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    if args.attack_type != "all":
        run_experiment(config, attack_type=args.attack_type)
        return

    # Phase 1: Single attack (baseline, no purification loss)
    print("\n### Phase 1: Single Attack (Baseline) ###\n")
    single_metrics, _ = run_experiment(config, attack_type="single")

    # Phase 2: Dual-target attack
    print("\n### Phase 2: Dual-Target Attack ###\n")
    dual_metrics, _ = run_experiment(config, attack_type="dual")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'Single':>10} {'Dual':>10}")
    print("-" * 60)
    for key in ["asr", "ppr", "purified_source_sim", "snr", "pesq", "stoi"]:
        print(f"{key:<30} {single_metrics[key]:>10.4f} {dual_metrics[key]:>10.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
