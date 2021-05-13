import importlib
import json
import os
from pathlib import Path
import pandas as pd


def load_class(full_class_string):
    """
    dynamically load a class from a string
    """
    # Replace for all possible OS path separators
    full_class_string = full_class_string.replace('/', '.')
    full_class_string = full_class_string.replace('\\', '.')

    class_data = full_class_string.split('.')
    module_path = '.'.join(class_data[:-1])
    class_str = class_data[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_str)


def read_input_samples_metadata(input_dir: str, input_models: list):
    """
    read input samples and metadata
    """
    input_samples = dict()
    input_metadatas = dict()
    # If model does not require any inputs, read "input_metadata.json" and "input_samples.json" to get necessary params
    if not input_models:
        input_models.append(
            {'model_type': 'input', 'model_name': 'input', 'model_id': 'input'})
    for input_model in input_models:
        model_name = input_model['model_name']
        input_prefix = Path(input_dir, model_name).as_posix()
        with open(input_prefix + '_samples.json') as f:
            input_samples[model_name] = json.load(f)
        with open(input_prefix + '_metadata.json') as f:
            input_metadatas[model_name] = json.load(f)

    # Check that all input metadata and samples are consistent
    assert(len(input_samples) > 0)
    assert(len(input_metadatas) > 0)
    t0_date = list(input_metadatas.values())[0]['t0_date']
    n_samples = list(input_metadatas.values())[0]['n_samples']
    dates_to_simulate = list(input_samples.values())[0]['dates']
    for input_metadata in input_metadatas.values():
        assert(input_metadata['t0_date'] == t0_date)
        assert(input_metadata['n_samples'] == n_samples)
    for input_sample in input_samples.values():
        assert(len(input_sample['samples']) == n_samples)
        assert(len(input_sample['dates']) == len(dates_to_simulate))
        for i in range(len(dates_to_simulate)):
            assert(input_sample['dates'][i] == dates_to_simulate[i])

    return (input_samples, input_metadatas)


def write_output_samples_metadata(output_dir: str, samples: dict, metadata: dict):
    """
   write output samples and metadata after model has been run
    """
    metadata_fname = Path(output_dir, 'output_metadata.json').as_posix()
    samples_fname = Path(output_dir, 'output_samples.json').as_posix()
    print('Writing file: ' + metadata_fname)
    print('Writing file: ' + samples_fname)
    with open(samples_fname, 'w') as f:
        json.dump(samples, f, indent=4)
    with open(metadata_fname, 'w') as f:
        json.dump(metadata, f, indent=4)


def get_model_schema(schema_dir: str):
    """
    retrieve schemas for config, metadata and samples
    """
    with open(os.path.join(schema_dir, 'config.schema')) as f:
        config_schema = json.load(f)
    with open(os.path.join(schema_dir, 'samples.schema')) as f:
        samples_schema = json.load(f)
    with open(os.path.join(schema_dir, 'metadata.schema')) as f:
        metadata_schema = json.load(f)
    return (config_schema, samples_schema, metadata_schema)


def split_data_by_zip_codes(fips_data: dict):
    """
    Takes values across a series of dates and the value's corresponding FIPS,
    and splits each value into the zip codes in its FIPS, proportional to the
    population of the zip code.

    Args:
        fips_data (dict of str: dict of str: int): data respresented
            by the mapping date -> fips -> value.
            Each fips should map to exactly one value.
    Returns:
        dict of str: dict of str: int: date -> zipcode -> value
    """
    fips_to_zip = pd.read_csv(
        Path('models/common', 'data', 'mit_state_fips_zip_with_population.csv').as_posix(),
        usecols=['zipcode', 'fips', 'population'],
        dtype={'zipcode': str, 'fips': str, 'population': 'Int64'})
    fips_to_zip.set_index('zipcode', inplace=True)

    output = dict()
    for date, fips_to_val in fips_data.items():
        output[date] = dict()
        for fips, val in fips_to_val.items():
            fips_df = fips_to_zip[fips_to_zip['fips'] == fips]
            zipcodes = fips_df.index.tolist()
            populations = [fips_df.loc[zipcode]['population'] for zipcode in zipcodes]
            total_pop = sum(populations)
            total_cases = val
            for zipcode, population in zip(zipcodes, populations):
                zip_cases = int(round(population / total_pop * total_cases))
                output[date][zipcode] = zip_cases
                total_pop = total_pop - population
                total_cases = total_cases - zip_cases
    return output


def normalize_to_one(counts_list: list):
    return [pi/sum(counts_list) for pi in list(counts_list)]


def day_of_week(date: str):
    day_num = pd.to_datetime(date).weekday()
    map = {0: 'Monday',
           1: 'Tuesday',
           2: 'Wednesday',
           3: 'Thursday',
           4: 'Friday',
           5: 'Saturday',
           6: 'Sunday'}
    return {'day_index': day_num,
            'day_name': map[day_num]}
