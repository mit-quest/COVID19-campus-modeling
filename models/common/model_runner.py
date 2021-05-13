import os
import time
from jsonschema import validate
from pathlib import Path

from models.common.model_utils import *


class ModelRunner():
    def run(self, io_dir: str, config_filepath: str, write_output=True, use_docker=False):
        if not use_docker:
            assert(config_filepath.startswith('models'))
        with open(config_filepath) as f:
            model_config = json.load(f)

        # Validate model_config
        if use_docker:
            schema_dir = os.path.join('/models', 'common', 'schema')
        else:
            repo_path = Path(os.path.abspath(__file__)).parents[2].as_posix()
            schema_dir = os.path.join(repo_path, 'models', 'common', 'schema')
        config_schema, samples_schema, metadata_schema = get_model_schema(
            schema_dir)
        validate(model_config, schema=config_schema)

        # Get inputs to model
        input_samples, input_metadatas = read_input_samples_metadata(
            io_dir, model_config['input_models'])
        t0_date = list(input_metadatas.values())[0]['t0_date']
        n_samples = list(input_metadatas.values())[0]['n_samples']
        dates_to_simulate = list(input_samples.values())[0]['dates']

        # Validate all input metadata and samples
        for input_metadata in input_metadatas.values():
            validate(input_metadata, schema=metadata_schema)
        for input_sample in input_samples.values():
            validate(input_sample, schema=samples_schema)

        # Load model module
        py_module_class_path = os.path.join(os.path.dirname(
            config_filepath), model_config['model_pyfile'], model_config['model_pyclass'])
        model = load_class(py_module_class_path)
        model_instance = model(model_config)

        # Sample from model
        # t0_date : str -> simulation start date
        # n_samples : int -> number of samples
        # dates_to_simulate : list of strings -> list of dates
        # input_samples : dict of dicts -> key: input model name, value: input samples (e.g. dates, samples)
        print('\nRunning model: ' + str(model_instance))
        tic = time.perf_counter()
        samples = model_instance.sample(
            t0_date, n_samples, dates_to_simulate, input_samples)
        time_elapsed = time.perf_counter()-tic
        print('Model runtime: ' + '{:.4f}'.format(time_elapsed) +
              's, Per sample: ' + '{:.4f}'.format(time_elapsed/n_samples) + 's')
        metadata = model_instance.config

        # Reset samples dates
        samples['dates'] = dates_to_simulate
        validate(samples, schema=samples_schema)

        metadata['t0_date'] = t0_date
        metadata['n_samples'] = n_samples
        assert(n_samples == len(samples['samples']))

        # Write to file
        if write_output:
            write_output_samples_metadata(io_dir, samples, metadata)
        return (samples, metadata)
