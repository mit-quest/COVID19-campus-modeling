import argparse
import copy
import json
import os
import time
from jsonschema import validate

from scheduler.common.scenario_runner import ScenarioRunner
from scheduler.common.api_utils import add_scenario_metadata, git_hash
from scheduler.common.batch_utils import get_scenario_schema

SCENARIOS_SCHEMA_DIR = os.path.join('scheduler', 'common', 'schema')
LOCAL_SCENARIOS_DIR = os.path.join('local_outputs', 'scenarios')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute scenarios')
    parser.add_argument('-s', '--use-server', dest='use_server', action='store_true',
                        help='whether to read / write samples & metadata to GCP bucket (default: false)')
    parser.add_argument('--scenario-fname', required=True, help='')
    parser.add_argument('-n', '--samples', dest='n_samples',
                        help='number of samples')
    parser.add_argument('-d', '--days-to-simulate',
                        help='number of days to simulate')
    parser.add_argument('-t', '--t0-date', help='t0 date')
    parser.add_argument('-u', '--uuid', help='Fix UUID prefix for debugging')
    parser.set_defaults(use_server=False)
    args = parser.parse_args()

    repo_path = os.path.dirname(os.path.abspath(__file__))
    with open(args.scenario_fname) as f:
        scenario_config = json.load(f)
    scenario_config_schema, scenario_metadata_schema = get_scenario_schema(os.path.join(repo_path, SCENARIOS_SCHEMA_DIR))
    validate(scenario_config, schema=scenario_config_schema)

    # commandline overrides
    if args.n_samples is not None:
        scenario_config['scenario_parameters']['n_samples'] = int(
            args.n_samples)
    if args.days_to_simulate is not None:
        scenario_config['scenario_parameters']['n_days_to_simulate'] = int(
            args.days_to_simulate)
    if args.t0_date is not None:
        scenario_config['scenario_parameters']['t0_date'] = args.t0_date
    if args.uuid is not None:
        uuid_prefix = args.uuid
    else:
        uuid_prefix = str(round(time.time()))
    runner = ScenarioRunner(scenario_config, uuid_prefix, args.use_server)
    runner.run()

    scenario_metadata = scenario_config

    # Write scenario metadata
    scenario_metadata['base_scenario_config'] = args.scenario_fname
    scenario_metadata['base_models'] = copy.deepcopy(scenario_config['models'])
    scenario_metadata.pop('models')
    scenario_metadata['base_analyses'] = copy.deepcopy(scenario_config['analyses'])
    scenario_metadata.pop('analyses')
    scenario_metadata['model_parameters'] = []
    scenario_metadata['analysis_parameters'] = []
    scenario_metadata['uuid_prefix'] = uuid_prefix
    scenario_metadata['git_hash'] = git_hash()
    scenario_metadata['batch_scenarios'] = [uuid_prefix]
    validate(scenario_metadata, schema=scenario_metadata_schema)

    print('\nWriting scenario metadata file')
    if not args.use_server:
        os.makedirs(os.path.join(repo_path, LOCAL_SCENARIOS_DIR), exist_ok=True)
    add_scenario_metadata(os.path.join(repo_path, LOCAL_SCENARIOS_DIR), uuid_prefix, scenario_metadata, args.use_server)
