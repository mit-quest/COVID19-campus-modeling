import base64
import git
import json
import os
import requests
import tempfile
import zlib
from google.cloud import secretmanager
from pathlib import Path
from requests.auth import HTTPBasicAuth

from . import zip_utils

# not needed in this release since UI runs on localhost
API_BASE = '<URL of your remote frontend UI server>'
ZIPJSON_KEY = 'base64(zip(o))'

# From https://medium.com/@busybus/zipjson-3ed15f8ea85d
# Example:
# original = {'a': "A", 'b': "B"}
# with open('test.zip', 'w') as f:
#       json.dump(json_zip(original), f, indent=4)
# with open('test.zip', 'r') as f:
#       unzipped = json_unzip(json.load(f)))


def json_unzip(zipped_json, insist=True):
    try:
        assert (zipped_json[ZIPJSON_KEY])
        assert (set(zipped_json.keys()) == {ZIPJSON_KEY})
    except:
        if insist:
            raise RuntimeError(
                'JSON not in the expected format {' + str(ZIPJSON_KEY) + ': zipstring}')
        else:
            return zipped_json

    try:
        unzipped_json = zlib.decompress(
            base64.b64decode(zipped_json[ZIPJSON_KEY]))
    except:
        raise RuntimeError('Could not decode/unzip the contents')

    try:
        loaded_json = json.loads(unzipped_json)
    except:
        raise RuntimeError('Could interpret the unzipped contents')

    return loaded_json


def json_zip(original_json):
    zipped_json = {
        ZIPJSON_KEY: base64.b64encode(
            zlib.compress(
                json.dumps(original_json).encode('utf-8')
            )
        ).decode('ascii')
    }
    return zipped_json


def git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def get_username_password_from_gcloud_secrets():
    username = 'quest'
    secrets_client = secretmanager.SecretManagerServiceClient()
    password_resource = f'projects/bridge-covid19/secrets/data-broker-password/versions/latest'
    password = secrets_client.access_secret_version(password_resource).payload.data.decode('UTF-8')
    return username, password


def get_samples_metadata_locally(local_dir: str, model_type: str, model_name: str, model_id: str, uuid: str) -> dict:
    input_prefix = Path(local_dir, model_type, model_name, model_id, uuid)

    with open(input_prefix.as_posix() + '_samples.json') as f:
        samples = json.load(f)
    with open(input_prefix.as_posix() + '_metadata.json') as f:
        metadata = json.load(f)
    return samples, metadata


def get_samples_metadata_from_server(uuid: str) -> dict:
    metadata_url = API_BASE + 'model/metadata/' + uuid
    print('Downloading: ' + metadata_url)
    user, passwd = get_username_password_from_gcloud_secrets()
    metadata_request = requests.get(metadata_url, auth=(user, passwd))

    if not metadata_request.ok:
        raise Exception(metadata_request.text)

    samples_url = API_BASE + 'model/samples/' + uuid
    print('Downloading: ' + samples_url)
    samples_request = requests.get(samples_url, auth=(user, passwd))

    if not samples_request.ok:
        raise Exception(samples_request.text)
    return json_unzip(samples_request.json()), json_unzip(metadata_request.json())


def get_samples_metadata(local_dir: str, uuid_prefix: str, t0_date: str, input_model: dict, use_server: bool) -> dict:
    input_uuid = '-'.join([uuid_prefix, input_model['model_name'], t0_date])
    if not use_server:
        return get_samples_metadata_locally(local_dir, input_model['model_type'], input_model['model_name'], input_model['model_id'], input_uuid)
    else:
        return get_samples_metadata_from_server(input_uuid)


def get_samples_metadata_from_scenario_config(local_dir: str, scenario_config_fname: str, model_name: str, uuid: str, use_server: bool) -> dict:
    # Load scenario config.json
    with open(scenario_config_fname) as f:
        scenario_config = json.load(f)

    model_config_desired = dict()
    for model_config_file in scenario_config['models']:
        with open(model_config_file) as f:
            model_config = json.load(f)
            if model_config['model_name'] == model_name:
                model_config_desired = model_config

    if not bool(model_config_desired):
        raise Exception('Model {} not found in {}'.format(model_name, scenario_config_fname))
    return get_samples_metadata(local_dir, uuid, scenario_config['scenario_parameters']['t0_date'], model_config_desired, use_server)


def add_samples_metadata_locally(local_dir: str, metadata: dict, samples: dict) -> bool:
    output_prefix = Path(local_dir, metadata['model_type'], metadata['model_name'],
                         metadata['model_id'], metadata['uuid']).as_posix()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    metadata_fname = output_prefix + '_metadata.json'
    samples_fname = output_prefix + '_samples.json'
    print('Storing file locally: ' + metadata_fname)
    print('Storing file locally: ' + samples_fname)

    with open(samples_fname, 'w') as f:
        json.dump(samples, f, indent=4)
    with open(metadata_fname, 'w') as f:
        json.dump(metadata, f, indent=4)

    return True


def add_samples_metadata_to_server(metadata: dict, samples: dict, allow_new_models: bool = True) -> bool:
    url = API_BASE + 'post/add-model-samples'
    payload = {'allow-new-models': 'True' if allow_new_models else ''}

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')
    samples_fd, samples_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(metadata, tmp, indent=4)
        with os.fdopen(samples_fd, 'w') as tmp:
            json.dump(zip_utils.json_zip(samples), tmp, indent=4)
        print('Uploading: ' + metadata['uuid'])
        r = requests.post(url, data=payload, files={'metadata': open(
            metadata_path, 'rb'), 'samples': open(samples_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)
        os.remove(samples_path)

    return True


def check_samples_metadata_with_server(metadata: dict, samples: dict) -> bool:
    url = API_BASE + 'post/check-model-samples-format'

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')
    samples_fd, samples_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(metadata, tmp, indent=4)
        with os.fdopen(samples_fd, 'w') as tmp:
            json.dump(zip_utils.json_zip(samples), tmp, indent=4)
        print('Checking format with server: ' + metadata['uuid'])
        r = requests.post(url, files={'metadata': open(
            metadata_path, 'rb'), 'samples': open(samples_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)
        os.remove(samples_path)

    return True


def get_analysis_metadata_locally(local_dir: str, analysis_name: str, analysis_id: str, uuid: str) -> dict:
    input_prefix = Path(local_dir, analysis_name, analysis_id, uuid)

    with open(input_prefix.as_posix() + '_results.json') as f:
        results = json.load(f)
    with open(input_prefix.as_posix() + '_metadata.json') as f:
        metadata = json.load(f)
    return results, metadata


def get_analysis_metadata_from_server(uuid: str) -> dict:
    metadata_url = API_BASE + 'analysis/metadata/' + uuid
    print('Downloading: ' + metadata_url)
    user, passwd = get_username_password_from_gcloud_secrets()
    metadata_request = requests.get(metadata_url, auth=(user, passwd))

    if not metadata_request.ok:
        raise Exception(metadata_request.text)

    results_url = API_BASE + 'analysis/results/' + uuid
    print('Downloading: ' + results_url)
    results_request = requests.get(results_url, auth=(user, passwd))

    if not results_request.ok:
        raise Exception(results_request.text)
    return json_unzip(results_request.json()), json_unzip(metadata_request.json())


def get_analysis_metadata(local_dir: str, uuid_prefix: str, t0_date: str, input_analysis: dict, use_server: bool) -> dict:
    input_uuid = '-'.join([uuid_prefix, input_analysis['analysis_name']])
    if not use_server:
        return get_analysis_metadata_locally(local_dir, input_analysis['analysis_name'], input_analysis['analysis_id'], input_uuid)
    else:
        return get_analysis_metadata_from_server(input_uuid)


def get_analysis_metadata_from_scenario_config(local_dir: str, scenario_config_fname: str, analysis_name: str, uuid: str, use_server: bool) -> dict:
    # Load scenario config.json
    with open(scenario_config_fname) as f:
        scenario_config = json.load(f)

    analysis_config_desired = dict()
    for analysis_config_file in scenario_config['analyses']:
        with open(analysis_config_file) as f:
            analysis_config = json.load(f)
            if analysis_config['analysis_name'] == analysis_name:
                analysis_config_desired = analysis_config

    if not bool(analysis_config_desired):
        raise Exception('Analysis {} not found in {}'.format(analysis_name, scenario_config_fname))
    return get_analysis_metadata(local_dir, uuid, scenario_config['scenario_parameters']['t0_date'], analysis_config_desired, use_server)


def get_analysis_results_from_sensitivity_config(local_dir: str, sensitivity_run_fname: str, analysis_name: str, use_server: bool) -> dict:
    # Load scenario config.json
    with open(sensitivity_run_fname) as f:
        sensitivity_config = json.load(f)

    analysis_configs_and_results = []

    for scenario_config in sensitivity_config['scenario_instance_configs']:
        latest_analysis_config_desired = dict()
        for analysis_config_file in scenario_config['analyses']:
            with open(analysis_config_file) as f:
                analysis_config = json.load(f)
                if analysis_config['analysis_name'] == analysis_name:
                    analysis_config_desired = analysis_config

        if not bool(analysis_config_desired):
            raise Exception('Analysis {} not found in {}'.format(analysis_name, scenario_config_fname))
        results, metadata = get_analysis_metadata(local_dir, scenario_config['within_batch_uuid'], scenario_config['scenario_parameters']['t0_date'],
                                                  analysis_config_desired, use_server)

        config_and_results = dict()
        config_and_results['scenario_config'] = scenario_config
        config_and_results['analysis_results'] = results
        config_and_results['analysis_metadata'] = metadata

        analysis_configs_and_results.append(config_and_results)
    return analysis_configs_and_results


def add_analysis_output_to_server(analysis_metadata: dict, analysis_results: dict, allow_new_models: bool = True) -> bool:
    url = API_BASE + 'post/add-analysis'
    payload = {'allow-new-analyses': 'True' if allow_new_models else ''}

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')
    results_fd, results_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(analysis_metadata, tmp, indent=4)
        with os.fdopen(results_fd, 'w') as tmp:
            json.dump(zip_utils.json_zip(analysis_results), tmp, indent=4)
        print('Uploading: ' + analysis_metadata['uuid'])
        r = requests.post(url, data=payload, files={'metadata': open(
            metadata_path, 'rb'), 'results': open(results_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)
        os.remove(results_path)

    return True


def add_analysis_output_locally(local_dir: str, analysis_metadata: dict, analysis_results: dict) -> bool:
    output_prefix = Path(local_dir, analysis_metadata['analysis_name'],
                         analysis_metadata['analysis_id'], analysis_metadata['uuid']).as_posix()
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    results_fname = output_prefix + '_results.json'
    metadata_fname = output_prefix + '_metadata.json'
    print('Storing file locally: ' + results_fname)
    print('Storing file locally: ' + metadata_fname)

    with open(results_fname, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    with open(metadata_fname, 'w') as f:
        json.dump(analysis_metadata, f, indent=4)

    return True


def add_scenario_metadata_to_server(scenario_metadata: dict) -> bool:
    url = API_BASE + 'post/add-scenario'

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(scenario_metadata, tmp, indent=4)
        print('Uploading: ' + scenario_metadata['uuid_prefix'])
        r = requests.post(url, files={'metadata': open(metadata_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)

    return True


def add_scenario_metadata_locally(local_dir: str, uuid_prefix: str, scenario_metadata: dict) -> bool:
    scenario_metadata_fname = os.path.join(
        local_dir, uuid_prefix + '_metadata.json')
    print('Storing file locally: ' + scenario_metadata_fname)
    with open(scenario_metadata_fname, 'w') as f:
        json.dump(scenario_metadata, f, indent=4)


def add_scenario_metadata(local_dir: str, uuid_prefix: str, scenario_metadata: dict, use_server: bool) -> dict:
    if not use_server:
        return add_scenario_metadata_locally(local_dir, uuid_prefix, scenario_metadata)
    else:
        return add_scenario_metadata_to_server(scenario_metadata)


def add_sensitivity_metadata_to_server(sensitivity_metadata: dict) -> bool:
    url = API_BASE + 'post/add-sensitivity'

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(sensitivity_metadata, tmp, indent=4)
        print('Uploading: ' + sensitivity_metadata['uuid_prefix'])
        r = requests.post(url, files={'metadata': open(metadata_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)

    return True


def add_sensitivity_metadata_locally(local_dir: str, uuid_prefix: str, sensitivity_metadata: dict) -> bool:
    sensitivity_metadata_fname = os.path.join(
        local_dir, uuid_prefix + '_metadata.json')
    print('Storing file locally: ' + sensitivity_metadata_fname)
    with open(sensitivity_metadata_fname, 'w') as f:
        json.dump(sensitivity_metadata, f, indent=4)


def add_sensitivity_metadata(local_dir: str, uuid_prefix: str, sensitivity_metadata: dict, use_server: bool) -> dict:
    if not use_server:
        return add_sensitivity_metadata_locally(local_dir, uuid_prefix, sensitivity_metadata)
    else:
        return add_sensitivity_metadata_to_server(sensitivity_metadata)


def add_historical_output_to_server(historical_metadata: dict, historical_results: dict, allow_new_historical_names: bool = True) -> bool:
    url = API_BASE + 'post/add-historical'
    payload = {'allow-new-historical-names': 'True' if allow_new_historical_names else ''}

    metadata_fd, metadata_path = tempfile.mkstemp(suffix='.json')
    results_fd, results_path = tempfile.mkstemp(suffix='.json')

    try:
        with os.fdopen(metadata_fd, 'w') as tmp:
            json.dump(historical_metadata, tmp, indent=4)
        with os.fdopen(results_fd, 'w') as tmp:
            json.dump(zip_utils.json_zip(historical_results), tmp, indent=4)
        print('Uploading: ' + historical_metadata['uuid'])
        r = requests.post(url, data=payload, files={'metadata': open(
            metadata_path, 'rb'), 'results': open(results_path, 'rb')})
        if not r.ok:
            raise Exception(r.text)
    finally:
        os.remove(metadata_path)
        os.remove(results_path)

    return True
