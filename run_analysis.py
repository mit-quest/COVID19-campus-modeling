import argparse
import json
import os
from jsonschema import validate
from pathlib import Path

from analyses.common.analysis_runner import AnalysisRunner
from scheduler.common.api_utils import get_samples_metadata, add_analysis_output_locally, git_hash, get_analysis_metadata

LOCAL_MODELS_DIR = os.path.join('local_outputs', 'models')
LOCAL_ANALYSES_DIR = os.path.join('local_outputs', 'analyses')
ANALYSES_SCHEMA_DIR = os.path.join('analyses', 'common', 'schema')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run individual analysis')
    parser.add_argument('--config-file', help='config filename', required=True)
    parser.add_argument(
        '--uuid-prefix', help='scenario uuid prefix (i.e. timestamp)', required=True)
    parser.add_argument('-s', '--use-server', dest='use_server', action='store_true',
                        help='whether to read / write to GCP bucket (default: false)')
    args = parser.parse_args()

    # TODO: Remove t0_date from uuid
    # t0_date = '2020-06-01'
    t0_date = '2020-08-01'
    # t0_date = '2020-06-15'
    use_server = args.use_server
    uuid_prefix = args.uuid_prefix
    config_filepath = args.config_file

    # Load analysis config.json
    with open(config_filepath) as f:
        analysis_config = json.load(f)

    all_input_samples = dict()
    for input_model in analysis_config['input_models']:
        input_samples, _ = get_samples_metadata(
            LOCAL_MODELS_DIR, uuid_prefix, t0_date, input_model, use_server)
        all_input_samples[input_model['model_type']] = input_samples

    all_input_analyses = dict()
    for input_analysis in analysis_config['input_analyses']:
        input_analysis_results, _ = get_analysis_metadata(
            LOCAL_ANALYSES_DIR, uuid_prefix, t0_date, input_analysis, use_server)
        all_input_analyses[input_analysis['analysis_name']] = input_analysis_results

    analysis = AnalysisRunner(uuid_prefix)
    analysis_results, analysis_metadata = analysis.run(
        config_filepath, all_input_samples, all_input_analyses)

    uuid = '-'.join([uuid_prefix, analysis_config['analysis_name']])
    analysis_metadata['uuid'] = uuid
    analysis_metadata['git_hash'] = git_hash()
    with open(os.path.join(ANALYSES_SCHEMA_DIR, 'metadata.schema')) as f:
        metadata_schema = json.load(f)
    validate(analysis_metadata, schema=metadata_schema)

    if not use_server:
        add_analysis_output_locally(
            LOCAL_ANALYSES_DIR, analysis_metadata, analysis_results)
    else:
        add_analysis_output_to_server(analysis_metadata, analysis_results)
