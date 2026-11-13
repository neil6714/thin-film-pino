"""Remote benchmarking pipeline comparing few-shot FNO and PINO on Colab GPU."""

import os
import subprocess
import sys
import time


def run_stream(cmd, desc):
    print(f"\n=======================================================", flush=True)
    print(f"=== [STAGE] {desc}", flush=True)
    print(f"=======================================================", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Stage '{desc}' failed with exit code {p.returncode}")
    print(f"=== {desc} completed in {time.time() - t0:.2f} s", flush=True)


def main():
    repo_dir = "/content/thin-film-pino"
    os.chdir(repo_dir)

    # Ensure repository is up to date
    run_stream(["git", "pull", "origin", "main"], "Git Pull Latest")

    os.makedirs("data/colab_run", exist_ok=True)
    os.makedirs("runs/colab_fewshot", exist_ok=True)

    # 1. Generate 256 trajectories
    run_stream(
        [
            sys.executable,
            "scripts/generate_data.py",
            "--mode",
            "trajectory",
            "--priors",
            "configs/priors_fast.yaml",
            "--num-samples",
            "256",
            "--batch-size",
            "32",
            "--num-cycles",
            "3",
            "--snapshots-per-phase",
            "8",
            "--nx",
            "40",
            "--ny",
            "30",
            "--device",
            "cuda",
            "--output",
            "data/colab_run/transient_ald_dataset.npz",
        ],
        "Generate 256 Trajectories on GPU",
    )

    # 2. Split dataset
    run_stream(
        [
            sys.executable,
            "scripts/prepare_temporal_dataset.py",
            "--input",
            "data/colab_run/transient_ald_dataset.npz",
            "--output",
            "data/colab_run/transient_ald_prepared.npz",
        ],
        "Stratified Dataset Splitting",
    )

    # 3. Train Few-Shot FNO (16 trajectories, 30 epochs)
    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.training.train",
            "--model",
            "fno",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--max-train-trajectories",
            "16",
            "--epochs",
            "30",
            "--batch-size",
            "4",
            "--seed",
            "42",
            "--output-dir",
            "runs/colab_fewshot/fno",
        ],
        "Train Few-Shot FNO (16 Trajectories)",
    )

    # 4. Train Few-Shot PINO (16 trajectories, 30 epochs, physics_weight=0.02, collocation_points=4)
    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.training.train",
            "--model",
            "pino",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--max-train-trajectories",
            "16",
            "--epochs",
            "30",
            "--batch-size",
            "4",
            "--physics-weight",
            "0.02",
            "--collocation-points",
            "4",
            "--seed",
            "42",
            "--output-dir",
            "runs/colab_fewshot/pino",
        ],
        "Train Few-Shot PINO (16 Trajectories)",
    )

    # 5. Evaluate Test Split Comparison
    run_stream(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--operator",
            "runs/colab_fewshot/fno/best_model.pt",
            "--pino",
            "runs/colab_fewshot/pino/best_model.pt",
            "--output",
            "runs/colab_fewshot/comparison_test",
            "--split",
            "test",
        ],
        "Compare Models on Test Set",
    )

    # 6. Evaluate OOD Split Comparison
    run_stream(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--operator",
            "runs/colab_fewshot/fno/best_model.pt",
            "--pino",
            "runs/colab_fewshot/pino/best_model.pt",
            "--output",
            "runs/colab_fewshot/comparison_ood",
            "--split",
            "ood",
        ],
        "Compare Models on OOD Set",
    )

    # 7. Evaluate Individual Model Metrics
    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.evaluation.evaluate",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--checkpoint",
            "runs/colab_fewshot/fno/best_model.pt",
            "--output-dir",
            "runs/colab_fewshot/eval_fno_test",
            "--split",
            "test",
        ],
        "Evaluate FNO on Test",
    )

    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.evaluation.evaluate",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--checkpoint",
            "runs/colab_fewshot/pino/best_model.pt",
            "--output-dir",
            "runs/colab_fewshot/eval_pino_test",
            "--split",
            "test",
        ],
        "Evaluate PINO on Test",
    )

    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.evaluation.evaluate",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--checkpoint",
            "runs/colab_fewshot/fno/best_model.pt",
            "--output-dir",
            "runs/colab_fewshot/eval_fno_ood",
            "--split",
            "ood",
        ],
        "Evaluate FNO on OOD",
    )

    run_stream(
        [
            sys.executable,
            "-m",
            "thin_film_pino.evaluation.evaluate",
            "--input",
            "data/colab_run/transient_ald_prepared.npz",
            "--checkpoint",
            "runs/colab_fewshot/pino/best_model.pt",
            "--output-dir",
            "runs/colab_fewshot/eval_pino_ood",
            "--split",
            "ood",
        ],
        "Evaluate PINO on OOD",
    )

    # 8. Plot loss trajectories
    run_stream(
        [
            sys.executable,
            "scripts/plot_losses.py",
            "--root",
            "runs/colab_fewshot",
            "--keys",
            "train_loss",
            "val_loss",
            "--output",
            "runs/colab_fewshot/loss_trajectories.png",
        ],
        "Plot Loss Trajectories",
    )

    # 9. Print Evaluation Summaries
    print("\n\n" + "=" * 80)
    print("=== TEST SET: FNO SUMMARY ===")
    print("=" * 80)
    with open("runs/colab_fewshot/eval_fno_test/summary.txt") as f:
        print(f.read())

    print("\n" + "=" * 80)
    print("=== TEST SET: PINO SUMMARY ===")
    print("=" * 80)
    with open("runs/colab_fewshot/eval_pino_test/summary.txt") as f:
        print(f.read())

    print("\n" + "=" * 80)
    print("=== OOD SET: FNO SUMMARY ===")
    print("=" * 80)
    with open("runs/colab_fewshot/eval_fno_ood/summary.txt") as f:
        print(f.read())

    print("\n" + "=" * 80)
    print("=== OOD SET: PINO SUMMARY ===")
    print("=" * 80)
    with open("runs/colab_fewshot/eval_pino_ood/summary.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
