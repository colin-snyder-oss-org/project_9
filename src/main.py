# src/main.py
import argparse
from src.training.train import train_model
from src.training.validate import validate_model
from src.deployment.edge_inference import run_edge_inference
from src.utils.config import Config

def main():
    parser = argparse.ArgumentParser(description='Sparse CNN Edge Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'inference'],
                        help='Mode to run the script in')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    config = Config(args.config)

    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'validate':
        validate_model(config)
    elif args.mode == 'inference':
        run_edge_inference(config)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
