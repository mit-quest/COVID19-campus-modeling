import argparse

from models.common.model_runner import ModelRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run individual models')
    parser.add_argument(
        '--io-dir', help='folder for input and output metadata and samples', required=True)
    parser.add_argument('--config-file', help='config filename', required=True)
    args = parser.parse_args()

    model_runner = ModelRunner()
    model_runner.run(args.io_dir, args.config_file)
