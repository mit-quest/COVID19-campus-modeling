import json
import os
from jsonschema import validate
from pathlib import Path

from models.common.model_utils import load_class
from scheduler.common import api_utils

ANALYSES_SCHEMA_DIR = os.path.join('analyses', 'common', 'schema')


class AnalysisRunner():
    def __init__(self, uuid_prefix: str):
        self.uuid_prefix = uuid_prefix

    def run(self, config_filepath: str, all_input_samples: dict, all_input_analyses: dict):
        with open(config_filepath) as f:
            analysis_config = json.load(f)

        assert(config_filepath.startswith('analyses'))
        pyfilepath = os.path.join(os.path.dirname(
            config_filepath), analysis_config['analysis_pyfile'])
        pyfilepath = pyfilepath.replace(os.sep, '.')
        py_module_class_path = pyfilepath + '.' + \
            analysis_config['analysis_pyclass']

        analysis = load_class(py_module_class_path)
        analysis_instance = analysis(analysis_config)

        with open(os.path.join(ANALYSES_SCHEMA_DIR, 'config.schema')) as f:
            config_schema = json.load(f)
        validate(analysis_config, schema=config_schema)

        print('\nRunning analysis: ' + str(analysis_instance))
        # input_samples : dict of dicts -> key: input model type, value: input samples (e.g. dates, samples)
        analysis_results = analysis_instance.run(all_input_samples, all_input_analyses, self.uuid_prefix)
        analysis_metadata = analysis_instance.config
        return (analysis_results, analysis_metadata)
