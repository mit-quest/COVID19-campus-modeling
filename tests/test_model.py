import glob
import json
import os
import unittest
from pathlib import Path

from models.common.model_runner import ModelRunner


class TestModels(unittest.TestCase):
    def setUp(self):
        self.model_runner = ModelRunner()

    def run_model(self, config_filename: str):
        # Use input samples & metadata from the relevant example_jsons folder
        io_dir = os.path.join(
            Path(config_filename).parents[1].as_posix(), 'example_jsons')

        with open(config_filename) as f:
            model_config = json.load(f)

        if 'run_unittest' in model_config and not model_config['run_unittest']:
            return

        if 'use_docker' in model_config and model_config['use_docker']:
            return

        (samples_one, metadata_one) = self.model_runner.run(
            io_dir, config_filename, False)
        (samples_two, metadata_two) = self.model_runner.run(
            io_dir, config_filename, False)

        self.assertEqual(json.dumps(samples_one), json.dumps(samples_two))
        self.assertEqual(json.dumps(metadata_one),
                         json.dumps(metadata_two))

    def test_all_models(self):
        """
        Test all models and ensure they run to completion
        Also test for reproducibility
        Ignores models that have the "use_docker" set to True
        """
        config_filenames = glob.glob('models/*/*/config.json')
        for config_filename in config_filenames:
            with self.subTest(config=config_filename):
                self.run_model(config_filename)


if __name__ == '__main__':
    unittest.main()
