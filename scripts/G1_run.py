import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.base_runner import Config, run_experiment

def main():
    cfg = Config(
        experiment_name="G1_CNN_BiLSTM_MSE",
        output_root="outputs/G1_CNN_BiLSTM_MSE",
        loss_type="mse",
        use_mcd=False
    )
    run_experiment(cfg, _PROJECT_ROOT)

if __name__ == "__main__":
    main()
