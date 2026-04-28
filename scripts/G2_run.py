import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.base_runner import Config, run_experiment

def main():
    cfg = Config(
        experiment_name="G2_CNN_BiLSTM_MSE_MCD",
        output_root="outputs/G2_CNN_BiLSTM_MSE_MCD",
        loss_type="mse",
        use_mcd=True,
        mc_samples_test=20
    )
    run_experiment(cfg, _PROJECT_ROOT)

if __name__ == "__main__":
    main()
