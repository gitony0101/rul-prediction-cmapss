import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.base_runner import Config, run_experiment

def main():
    cfg = Config(
        experiment_name="G3_CNN_BiLSTM_LinEx",
        output_root="outputs/G3_CNN_BiLSTM_LinEx",
        loss_type="linex",
        linex_a=0.04,
        use_mcd=False
    )
    run_experiment(cfg, _PROJECT_ROOT)

if __name__ == "__main__":
    main()
