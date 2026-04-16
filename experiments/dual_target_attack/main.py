"""
Main experiment script for dual-target adversarial attack
"""

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
        model_path=config.speaker_model_path, device=config.device
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
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Initialize metrics
    metrics_evaluator = AttackMetrics(speaker_model, purification_model, config.device)

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
        all_metrics.append(batch_metrics)

        # Save audio samples for listening
        audio_dir = os.path.join(config.log_dir, f"audio_{attack_type}_{run_ts}")
        os.makedirs(audio_dir, exist_ok=True)
        with torch.no_grad():
            x_purified = metrics_evaluator.purification_model(x_adv)

        def save_wav(path, tensor):
            arr = (tensor.squeeze().cpu().numpy() * 32767).astype(np.int16)
            wavfile.write(path, config.sample_rate, arr)

        for i in range(x_clean.shape[0]):
            idx = batch_idx * config.batch_size + i
            if idx % 4 != 0:
                continue
            save_wav(f"{audio_dir}/{idx:03d}_clean.wav", x_clean[i])
            save_wav(f"{audio_dir}/{idx:03d}_adv.wav", x_adv[i])
            save_wav(f"{audio_dir}/{idx:03d}_purified.wav", x_purified[i].squeeze(0))

        # Log progress and checkpoint
        if batch_idx % config.log_interval == 0:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{ts}] Batch {batch_idx}: ASR={batch_metrics['asr']:.2%}, "
                f"PPR={batch_metrics['ppr']:.2%}, "
                f"PurifiedSim(target)={batch_metrics['purified_target_sim']:.4f}, "
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
        key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()
    }

    metrics_evaluator.print_metrics(avg_metrics)

    # Save results
    results = {
        "attack_type": attack_type,
        "config": {
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
    print("\n" + "=" * 60)
    print("Dual-Target Adversarial Attack Experiment")
    print("=" * 60 + "\n")

    # Create output directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)

    # Phase 1: Single attack (baseline, no purification loss)
    print("\n### Phase 1: Single Attack (Baseline) ###\n")
    single_metrics, single_success = run_experiment(config, attack_type="single")

    # Phase 2: Dual-target attack
    print("\n### Phase 2: Dual-Target Attack ###\n")
    dual_metrics, dual_success = run_experiment(config, attack_type="dual")

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
