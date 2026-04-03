from pathlib import Path
import csv
import matplotlib.pyplot as plt


INPUT_CSV = Path("outputs_v2/group_comparison.csv")
OUTPUT_PNG = Path("figures/group_rmse_comparison.png")


def load_rmse_data(csv_path: Path):
    groups = []
    rmse_means = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups.append(row["group"])
            rmse_means.append(float(row["rmse_mean"]))

    return groups, rmse_means


def main():
    groups, rmse_means = load_rmse_data(INPUT_CSV)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(groups, rmse_means)

    plt.xlabel("Experiment Group")
    plt.ylabel("Mean RMSE")
    plt.title("RUL Prediction Group Comparison (Mean RMSE)")
    plt.ylim(min(rmse_means) - 0.5, max(rmse_means) + 0.5)

    for bar, value in zip(bars, rmse_means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved figure to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
