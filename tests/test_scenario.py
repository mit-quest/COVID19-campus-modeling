import glob
import json
import os
import unittest
from jsonschema import validate
from pathlib import Path

from scheduler.common.scenario_runner import ScenarioRunner

LOCAL_OUTPUTS_DIR = 'local_outputs'
SCENARIOS_SCHEMA_DIR = os.path.join('scheduler', 'common', 'schema')


class TestScenarios(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.repo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..')

    def run_scenario(self, scenario_fname: str, scenario_schema: dict):
        with open(scenario_fname) as f:
            scenario_config = json.load(f)
        validate(scenario_config, schema=scenario_schema)

        scenario_params = scenario_config['scenario_parameters']
        if 'run_unittest' in scenario_params and not scenario_params['run_unittest']:
            return

        # Reduce number of samples and dates to speed up integration tests
        scenario_config['scenario_parameters']['n_samples'] = 10
        scenario_config['scenario_parameters']['n_days_to_simulate'] = 3

        scenario_name = os.path.splitext(os.path.basename(scenario_fname))[0]

        runner_one = ScenarioRunner(
            scenario_config, 'test1-' + scenario_name, False)
        runner_one.run()
        runner_two = ScenarioRunner(
            scenario_config, 'test2-' + scenario_name, False)
        runner_two.run()

        for path, subdirs, files in os.walk(os.path.join(self.repo_path, LOCAL_OUTPUTS_DIR)):
            for filename in files:
                # Only compare model samples and analysis results jsons
                if not filename.startswith('test1-' + scenario_name):
                    continue
                if not filename.endswith('samples.json') and not filename.endswith('results.json'):
                    continue
                test1_fname = os.path.join(path, filename)
                test2_fname = test1_fname.replace('test1', 'test2')

                print('Comparing: ' + test1_fname)
                with open(test1_fname) as f:
                    test1_json = json.load(f)
                with open(test2_fname) as f:
                    test2_json = json.load(f)
                self.assertEqual(json.dumps(test1_json),
                                 json.dumps(test2_json))

    def test_all_scenarios(self):
        """
        Test that there is no randomness in the models, and results are reproducible
        Runs the same scheduler twice, and compares all the output sample files

        If this test fails, it is likely that the models used for the scenario have a random number generator that has not specified a static seed.
        """
        scenario_filelist = glob.glob('scheduler/scenarios/*.json')

        with open(os.path.join(self.repo_path, SCENARIOS_SCHEMA_DIR, 'scenario_config.schema')) as f:
            scenario_schema = json.load(f)

        for scenario_fname in scenario_filelist:
            with self.subTest(scenario=scenario_fname):
                self.run_scenario(scenario_fname, scenario_schema)


if __name__ == '__main__':
    unittest.main()
