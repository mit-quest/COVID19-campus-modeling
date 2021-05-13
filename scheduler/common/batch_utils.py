import json
import os
import tempfile


def get_scenario_schema(schema_dir: str) -> (dict, dict):
    """
    retrieve schemas for config and metadata
    """
    with open(os.path.join(schema_dir, 'scenario_config.schema')) as f:
        config_schema = json.load(f)
    with open(os.path.join(schema_dir, 'scenario_metadata.schema')) as f:
        metadata_schema = json.load(f)
    return (config_schema, metadata_schema)


def override_scenario_params(base_scenario_config: dict, scenario_params_to_change: [dict]) -> dict:
    """
    overrides the scenario parameters in the base scenario config
    """
    for name_value in scenario_params_to_change:
        param_name = name_value['name']
        param_value = name_value['value']
        print("Overriding '" + param_name + "' -> " + str(param_value))
        base_scenario_config['scenario_parameters'][param_name] = param_value

    return base_scenario_config


def generate_tmp_model_configs(model_configs: [dict], base_config_fnames: [str], repo_root: str) -> [str]:
    """
    Create temp versions of the model config files in the same folder as the original model config files
    The temp versions typically have slightly varied model parameters from the original model config files
    """
    tmp_config_filepaths = []
    for i, model_config in enumerate(model_configs):
        model_dir = os.path.dirname(base_config_fnames[i])
        tmpfile = tempfile.NamedTemporaryFile(dir=model_dir)
        tmpfile_partition = tmpfile.name.partition('models/')
        tmp_filepath = os.path.join(
            tmpfile_partition[1], tmpfile_partition[2] + '.json')
        with open(os.path.join(repo_root, tmp_filepath), 'w') as f:
            json.dump(model_config, f, indent=4)
        tmp_config_filepaths.append(tmp_filepath)
    return tmp_config_filepaths
