import datetime
import glob
import pandas as pd
import subprocess
import tempfile
from jsonschema import validate

from analyses.common.analysis_runner import AnalysisRunner
from models.common.model_runner import ModelRunner
from models.common.model_utils import get_model_schema
from scheduler.common.api_utils import *

METADATA_SUFFIX = '_metadata.json'
SAMPLES_SUFFIX = '_samples.json'
LOCAL_ANALYSES_DIR = os.path.join('local_outputs', 'analyses')
LOCAL_MODELS_DIR = os.path.join('local_outputs', 'models')
MODELS_SCHEMA_DIR = os.path.join('models', 'common', 'schema')
ANALYSES_SCHEMA_DIR = os.path.join('analyses', 'common', 'schema')


class ScenarioRunner():
    # Get repo root path
    repo_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', '..')

    def __init__(self, scenario_config: dict, uuid_prefix: str, use_server: bool):
        self.scenario_config = scenario_config
        self.scenario_params = scenario_config['scenario_parameters']
        self.uuid_prefix = uuid_prefix
        self.use_server = use_server

        sim_start_date = pd.to_datetime(self.scenario_params['t0_date'])
        sim_end_date = pd.to_datetime(self.scenario_params['t0_date']) + datetime.timedelta(
            days=self.scenario_params['n_days_to_simulate'])
        self.dates = [d.strftime('%Y-%m-%d')
                      for d in pd.date_range(sim_start_date, sim_end_date)]

    def create_samples_metadata_without_input_models(self, t0_date: str, n_samples: int, dates: list):
        samples = []
        for _ in range(n_samples):
            samples.append(list())
        input_samples = dict()
        input_samples['dates'] = dates
        input_samples['samples'] = samples

        input_metadata = {'model_type': '', 'model_name': '', 'model_id': '', 'uuid': '',
                          't0_date': t0_date, 'n_samples': n_samples, 'input_models': [], 'model_parameters': dict(),
                          'git_hash': ''}
        return (input_samples, input_metadata)

    def write_input_model_to_tempdir(self, input_samples: dict, input_metadata: dict, input_prefix: str):
        with open(input_prefix + SAMPLES_SUFFIX, 'w') as f:
            json.dump(input_samples, f, indent=4)
        with open(input_prefix + METADATA_SUFFIX, 'w') as f:
            json.dump(input_metadata, f, indent=4)

    def run_model(self, config_path: str):
        t0_date = self.scenario_params['t0_date']
        n_samples = self.scenario_params['n_samples']

        # Load model config.json
        with open(config_path) as f:
            model_config = json.load(f)
        config_schema, samples_schema, metadata_schema = get_model_schema(
            os.path.join(self.repo_path, MODELS_SCHEMA_DIR))
        validate(model_config, schema=config_schema)

        model_type = model_config['model_type']
        model_name = model_config['model_name']
        input_models = model_config['input_models']
        uuid = '-'.join([self.uuid_prefix, model_name, t0_date])

        models_tmp_dir = os.path.join(self.repo_path, 'models', 'tmp')
        os.makedirs(models_tmp_dir, exist_ok=True)
        tempdir = tempfile.TemporaryDirectory(dir=models_tmp_dir)
        Path(tempdir.name).mkdir(exist_ok=True)

        docker_model = False
        if 'use_docker' in model_config:
            docker_model = model_config['use_docker']

        # Get input metadata and samples for model (locally or from server)
        if not any(input_models):
            # Create input metadata and samples with params for models w/o input
            input_samples, input_metadata = self.create_samples_metadata_without_input_models(
                t0_date, n_samples, self.dates)
            input_prefix = Path(tempdir.name, 'input').as_posix()
            validate(input_metadata, schema=metadata_schema)
            validate(input_samples, schema=samples_schema)
            self.write_input_model_to_tempdir(
                input_samples, input_metadata, input_prefix)
        else:
            for input_model in input_models:
                input_samples, input_metadata = get_samples_metadata(
                    LOCAL_MODELS_DIR, self.uuid_prefix, t0_date, input_model, self.use_server)
                validate(input_metadata, schema=metadata_schema)
                # Write to input files to tempdir
                input_prefix = Path(
                    tempdir.name, input_model['model_name']).as_posix()
                self.write_input_model_to_tempdir(
                    input_samples, input_metadata, input_prefix)

        # Run model, generate samples and metadata
        if not docker_model:
            model = ModelRunner()
            model.run(tempdir.name, config_path)
        else:
            print('Building docker image: ' + model_name)
            # docker needs to be run from <repo_path>/models
            model_dir = os.path.dirname(config_path)
            docker_fname_from_models_dir = os.path.join(
                model_dir[len('models/'):], 'Dockerfile')
            cmd_list = ['docker', 'build', '-t', model_name, '.', '-f', docker_fname_from_models_dir,
                        '--build-arg', 'model_dir=' + os.path.dirname(docker_fname_from_models_dir)]
            subprocess.run(cmd_list, cwd=os.path.join(
                self.repo_path, 'models'))
            cmd_list = ['docker', 'run', '-v',
                        tempdir.name + ':/io_dir', model_name]
            subprocess.run(cmd_list)

        # Retrieve output files from temp dir
        output_prefix = Path(tempdir.name, 'output').as_posix()
        with open(output_prefix + SAMPLES_SUFFIX) as f:
            output_samples = json.load(f)
        with open(output_prefix + METADATA_SUFFIX) as f:
            output_metadata = json.load(f)
        output_metadata['uuid'] = uuid
        output_metadata['git_hash'] = git_hash()
        for input_model in output_metadata['input_models']:
            input_model['uuid'] = '-'.join(
                [self.uuid_prefix, input_model['model_name'], output_metadata['t0_date']])

        # Validate output metadata and schema
        validate(output_metadata, schema=metadata_schema)
        validate(output_samples, schema=samples_schema)
        assert(output_samples['dates'] == self.dates)

        # Store output samples and metadata (locally or from server)
        if not self.use_server:
            add_samples_metadata_locally(
                LOCAL_MODELS_DIR, output_metadata, output_samples)
        else:
            add_samples_metadata_to_server(output_metadata, output_samples)

        # Clear tempdir
        tempfiles = glob.glob(os.path.join(tempdir.name, '*'))
        for f in tempfiles:
            os.remove(f)
        tempdir.cleanup()

    def run_analysis(self, config_path: str):
        t0_date = self.scenario_params['t0_date']
        n_samples = self.scenario_params['n_samples']

        # Load analysis config.json
        with open(config_path) as f:
            analysis_config = json.load(f)

        with open(os.path.join(ANALYSES_SCHEMA_DIR, 'config.schema')) as f:
            config_schema = json.load(f)
        validate(analysis_config, schema=config_schema)

        analysis_name = analysis_config['analysis_name']
        uuid = '-'.join([self.uuid_prefix, analysis_name])

        all_input_samples = dict()
        for input_model in analysis_config['input_models']:
            input_samples, _ = get_samples_metadata(
                LOCAL_MODELS_DIR, self.uuid_prefix, t0_date, input_model, self.use_server)
            all_input_samples[input_model['model_type']] = input_samples

        all_input_analyses = dict()
        for input_analysis in analysis_config['input_analyses']:
            input_analysis_results, _ = get_analysis_metadata(
                LOCAL_ANALYSES_DIR, self.uuid_prefix, t0_date, input_analysis, self.use_server)
            all_input_analyses[input_analysis['analysis_name']] = input_analysis_results

        analysis = AnalysisRunner(self.uuid_prefix)
        analysis_results, analysis_metadata = analysis.run(
            config_path, all_input_samples, all_input_analyses)

        analysis_metadata['uuid'] = uuid
        analysis_metadata['git_hash'] = git_hash()
        with open(os.path.join(ANALYSES_SCHEMA_DIR, 'metadata.schema')) as f:
            metadata_schema = json.load(f)
        validate(analysis_metadata, schema=metadata_schema)

        if not self.use_server:
            add_analysis_output_locally(
                LOCAL_ANALYSES_DIR, analysis_metadata, analysis_results)
        else:
            add_analysis_output_to_server(analysis_metadata, analysis_results)

    def run(self):
        for model_config_path in self.scenario_config['models']:
            self.run_model(model_config_path)

        for analysis_config_path in self.scenario_config['analyses']:
            self.run_analysis(analysis_config_path)
