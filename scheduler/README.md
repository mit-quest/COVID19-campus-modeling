# MIT COVID-19 Risk Model (RM)


## Running a scenario

A scenario runs multiple models and analyses via a `scenario_<name>.json` file

It also orchestrates the passing of the `samples` and `metadata` json files between models and analyses

To run a scenario
```sh
$ python run_scenario.py --scenario-fname <scenario_filename>
```
For example, to run the placeholder scenario file:
```sh
$ python run_scenario.py --scenario-fname scheduler/scenarios/risk_model_simulation_configuration.json
```
Options for running scenarios:
- `-s`: If specified, scheduler reads / writes files to the simulations server / data broker. Otherwise, files are written to local folder
- `-n`: Overrides the no. of samples specified in the scenario file
- `-d`: Overrides the no. of days to simulate in the scenario file
- `-t` : Overrides the `t0_date` in the scenario file

To add a new model / analysis to a scenario, add the model / analysis `config.json` file location to the `scheduler/scenarios/<scenario_name>.json` file, in the desired run sequence


## Running a model
To run a model:
```sh
$ python run_model.py --io-dir <io_directory> --config-file <config_file>
```
The model's output will then be written to the following files:
```
<io_directory>/output_metadata.json
<io_directory>/output_samples.json
```
For example, to run the action placeholder model:
```sh
$ python run_model.py --io-dir models/action/example_jsons --config-file models/action/placeholder/config.json
```
To create a new model, please refer to the [models README](models/README.md)


## Running an analysis

Analyses are scripts that help compute statistics of a model / scenario run, e.g. population statistics

It assumes that a scenario run has been completed successfully, and that the model output samples have been written succssfully to the local folder or the simulations server

To run an analysis:
```sh
$ python run_analysis.py --config-file <config_file> --uuid-prefix <uuid_prefix>
```
Options
- `-s`: If specified, analysis input / outputs are written to and obtained from the simulations server / data broker. Otherwise, files are written to local folder

If run locally, the analysis output will be written to:
```
local_outputs/analyses/<analysis_name>/<analysis_id>/<uuid_prefix>-<analysis_name>_results.json
local_outputs/analyses/<analysis_name>/<analysis_id>/<uuid_prefix>-<analysis_name>_metadata.json
```

To create a new analysis, please refer to the [analysis README](analyses/README.md)


## Unit and Integration Tests

Before submitting / merging a PR, it is important to confirm that all unit and integration tests still pass
```sh
$ python -m unittest tests.test_model.TestModels
$ python -m unittest tests.test_scenario.TestScenarios
```

These tests are also run as pre-merge Github Actions, and will prevent you from merging a PR if they fail

## Creating new scenarios
Create a new JSON with in `scheduler/scenarios`
```sh
$ cd scheduler/scenarios
```
Copy the contents of the existing config `risk_model_simulation_configuration.json`
```sh
$ cp risk_model_simulation_configuration.json <scenario_name>
```
Modify the parameters in `<scenario_name>`:
- `"n_samples": 1000` indicates that the number of Monte Carlo samples for each model will be 1000
- `"t0_date": "2020-08-01"` indicates that the starting date of the simulation is Aug 1 2020
- `"n_days_to_simulate": 7` indicates that the number of dates to simulate is 7
- `"models"` list of models to run in this simulation. This assume that each model is being supplied the correct previous model's data, as defined in each model's `config.json` file.
- `"analysis"` list of analysis to run in this simulation. This assume that each analysis is being supplied the correct model's data, as defined in each analysis' `config.json` file.

