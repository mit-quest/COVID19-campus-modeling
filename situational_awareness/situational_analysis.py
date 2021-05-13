import os
import re
import sys
import time
import json
import copy
import tqdm
import datetime
import warnings
import shutil
import random
import cProfile
import functools
import argparse
import logging
import gzip


import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.dates as mdates
import multiprocessing

from math import isnan
from google.cloud import storage
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.dates import DateFormatter
from pathlib import Path

# so we can import models below from top level directory
# note this script should be run from top level directory e.g.
# python situational_awareness/situational_analysis.py --config-file situational_awareness/config.json
sys.path.append(str(Path('.').absolute()))
from models.common.to_precision import sig_fig_formatted  # noqa: E402
from analyses.common.analysis import Analysis, sort_dict  # noqa: E402
from models.common.mit_buildings import MITBuildings  # noqa: E402


# # if needed down the line for easier reading/editing of matplotlib pdf plot files for presentations
# mpl.rcParams['pdf.fonttype'] = 42  # noqa: E402


def stringify(*args):
    """
    little function to convert many inputs to string
    so i can easily replace print by logging
    """
    output_string = ''
    for chunk in args:
        output_string = output_string + ' ' + str(chunk)
    return output_string[1:]


def convert_timestamp_to_hour_of_week(timestamp):
    return int(timestamp.weekday() * 24 + timestamp.hour)


def profile(func):
    # magical incantation from https://esmithy.net/2015/12/02/granular-profiling-in-pycharm/
    # to output a .pstat file for profiling a specific function
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            ret = func(*args, **kwargs)
            profiler.disable()
            return ret
        finally:
            filename = os.path.expanduser(
                os.path.join('./', func.__name__ + '.pstat')
            )
            profiler.dump_stats(filename)

    return wrapper


class SituationalAwareness:

    def __init__(self, config_filepath: str):
        self.uuid_prefix = str(round(time.time()))

        # Load analysis config.json
        with open(config_filepath) as f:
            simulation_config = json.load(f)

        self.config = simulation_config
        self.config['execution_time'] = self.uuid_prefix
        self.config['config_filname_run'] = config_filepath
        self.experiment_name = self.config['simulation_experiment_name']

        self.seed = self.config['simulation_parameters']['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.log_file_location = self.config['simulation_parameters']['log_file_location']
        self.setup_logger(
            self.log_file_location,
            self.uuid_prefix,
            self.experiment_name
        )

        cpu_count = multiprocessing.cpu_count()
        self.logger.info(stringify('CPU count on machine:', cpu_count))
        self.PLOT = self.config['simulation_parameters']['plot']
        # self.DEBUG_MODE = self.config['simulation_parameters']['debug']
        self.limited_buiding_list = self.config['simulation_parameters']['limited_buiding_list']
        self.GROUND_TRUTHING = self.config['simulation_parameters']['ground_truthing']
        self.fit_n_hours_window_backward = self.config['simulation_parameters']['fit_n_hours_window_backward']
        self.simulate_n_hours_window_backward = self.config['simulation_parameters']['simulate_n_hours_window_backward']
        self.simulate_n_hours_window_forward = self.config['simulation_parameters']['simulate_n_hours_window_forward']
        self.plot_n_hours_window_backward = self.config['simulation_parameters']['plot_n_hours_window_backward']
        self.plot_n_hours_window_forward = self.config['simulation_parameters']['plot_n_hours_window_forward']

        if self.simulate_n_hours_window_backward >= self.fit_n_hours_window_backward:
            error_message = "error: you don't have enough data ({}) to simulate the future ({})".format(
                self.fit_n_hours_window_backward, self.simulate_n_hours_window_backward)
            self.logger.exception(stringify(error_message))
            raise ValueError(error_message)

        self.N_REPLICATIONS = self.config['simulation_parameters']['n_replications']
        self.percentiles_list = self.config['simulation_parameters']['percentiles_list']
        self.debug_cardreader_data = self.config['simulation_parameters']['data_parameters']['cardreader_data']
        self.debug_cardreader_metadata = self.config['simulation_parameters']['data_parameters']['cardreader_metadata']
        self.cardreader_key_name = self.config['simulation_parameters']['data_parameters']['cardreader_key_name']
        self.debug_building_list = self.config['simulation_parameters']['debug_building_list']
        self.num_trials_assignments = self.config['simulation_parameters']['num_trials_assignments']
        self.num_trials_transitions = self.config['simulation_parameters']['num_trials_transitions']
        # TODO: add timestamp to save plot location https://www.geeksforgeeks.org/get-current-timestamp-using-python/
        self.plot_location = self.config['simulation_parameters']['plot_location'] + '_' + self.experiment_name
        self.inflow_model = self.empirical_situational_inflow_model
        self.occupancy_model = self.empirical_situational_occupancy_model

        self.all_buildings = MITBuildings().known_building_ids()

        building_transition_function_name = self.config['simulation_parameters']['building_transition_function']
        # is there a less hacky way to do this?
        self.building_transition_function = getattr(self, building_transition_function_name)
        # exec('self.building_transition_function = self.{}'.format(building_transition_function_name))

        self.holiday_list = self.make_holidays_list()

        self.inject_in_person_class_headcount = self.config['simulation_parameters']['inject_in_person_class_headcount']

        self.io_dir = self.config['simulation_parameters']['io_dir']
        Path(self.io_dir).mkdir(parents=True, exist_ok=True)

        # to remove pyro RuntimeWarning trying to observe value outside inference
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.logger.info(stringify('Simulation variant:', self.experiment_name))

    def run(self) -> dict:
        global_start = time.time()

        ###############
        # reading card reader data and setting up simulation variables
        ###############
        self.manage_card_reader()

        ###############
        # reading schedule injection data
        ###############
        self.schedule_injection, self.student_buildings, self.building_deltas = self.read_schedule_injection()

        ###############
        # reading COVID-pass assignments
        ###############
        self.all_buildings_assignments_pmf = self.make_assignment_buildings_count()

        ###############
        # reading wifi transitions
        ###############
        self.all_buildings_transition_pmf = self.make_buildings_transition_pmf()

        ###############
        # reading wifi durations
        ###############
        self.duration_pmf_dict, self.mean_duration, self.std_duration = self.make_durations_pmf_dict()

        ###############
        # running simulations
        ###############
        self.make_output_past_future_dict()

        ###############
        # outputing simulated data
        ###############
        self.output_json = self.make_output_json()

        ###############
        # ground truth comparison
        ###############
        if self.GROUND_TRUTHING:
            self.create_ground_truth_comparison()

        ###############
        # plotting
        ###############
        if self.PLOT:
            self.plot_simulated_timeseries()
            self.report_percent_non_naive_transitions()
            self.plot_observed_building_hourly_arrivals()
            self.plot_simulated_building_to_building_histogram()
            self.render_simulated_campus_hourly_plot('inflow', 'outflow', 'occupancy', 'per_hour')

        ##############
        # saving output json and metadata to disk
        # ##############
        self.write_output_n_metadata_to_disk()

        self.logger.info(stringify('script took {} seconds'.format(time.time() - global_start)))

    def setup_logger(self,
                     log_file_location: str,
                     uuid: str,
                     experiment_name: str
                     ):

        os.makedirs(log_file_location, exist_ok=True)
        self.log_file = '{}-{}.log'.format(uuid, experiment_name)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        logFormatter = logging.Formatter(
            '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s', '%Y-%m-%d %H:%M:%S')

        # to ouput log to both console (terminal) and log file
        output_file_handler = logging.FileHandler(os.path.join(log_file_location,
                                                               self.log_file)
                                                  )
        output_file_handler.setFormatter(logFormatter)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logFormatter)

        self.logger.addHandler(output_file_handler)
        self.logger.addHandler(stdout_handler)

        self.logger.info('Folder {} created.'.format(log_file_location))
        self.logger.info('Log File {} created.'.format(
            os.path.join(log_file_location,
                         self.log_file)
        )
        )

        """
        example of what logger outputs to both console and terminal:

        2021-03-19 15:28 [MainThread  ] [INFO ]  Folder situational_awareness/ created.
        2021-03-19 15:28 [MainThread  ] [INFO ]  Log File situational_awareness/1616167717-own_analysis_class.log created.
        2021-03-19 15:28 [MainThread  ] [INFO ]  CPU count on machine: 96
        2021-03-19 15:28 [MainThread  ] [INFO ]  Simulation variant: own_analysis_class
        2021-03-19 15:28 [MainThread  ] [INFO ]  [read_cardreader_json_to_dict] reading card reader json
        2021-03-19 15:28 [MainThread  ] [INFO ]  [read_cardreader_json_to_dict] done in: 1.1127948760986328
        2021-03-19 15:28 [MainThread  ] [INFO ]  [transform_cardreader_json_dict_to_pandas] converting json to pandas dataframe
        """

    def manage_card_reader(self):
        self.card_reader_historical_dict, self.ingestion_list_buildings, self.end_of_backwards_window, self.earliest_time_from_card_reader_data \
            = self.read_cardreader_json_to_dict()

        if self.GROUND_TRUTHING:
            # if this is not a ground truthing run, then this does not over-write previously defined self.end_of_backwards_window
            self.end_of_backwards_window = pd.to_datetime(
                self.config['data_parameters']['ground_truth_end_of_backwards_window'])

            ground_truth_filename = self.config['simulation_parameters']['data_parameters']['ground_truth_day']
            with open(ground_truth_filename) as fp:
                self.ground_truth_dict = json.load(fp)

            self.buildings_under_scrutiny = list(self.ground_truth_dict.keys())
            self.ground_truth_data_range = list(self.ground_truth_dict[list(
                self.ground_truth_dict.keys())[0]].keys())[0].split(' - ')
            self.ground_truth_data_start = pd.to_datetime(self.ground_truth_data_range[0])
            self.ground_truth_data_end = pd.to_datetime(self.ground_truth_data_range[1])

        """
        time specifications for start+end of backward+forward windows of:
        - fitting
        - simulation
        - plotting
        """
        # fitting
        self.fitting_beginning_of_backwards_window = self.end_of_backwards_window - \
            timedelta(hours=self.fit_n_hours_window_backward)

        if self.earliest_time_from_card_reader_data \
                >= self.fitting_beginning_of_backwards_window:
            error_message = "Data's earliest date {} is not earlier than beginning of simulation {}".format(
                self.earliest_time_from_card_reader_data, self.fitting_beginning_of_backwards_window)
            self.logger.exception(stringify(error_message))
            raise ValueError(error_message)

        # simulation
        self.simulation_beginning_of_backwards_window = self.end_of_backwards_window - \
            timedelta(hours=self.simulate_n_hours_window_backward)
        self.simulation_beginning_of_future_window = self.end_of_backwards_window + timedelta(hours=1)
        self.simulation_end_of_future_window = self.simulation_beginning_of_future_window + \
            timedelta(hours=self.simulate_n_hours_window_forward)
        self.simulation_full_time_window = pd.date_range(self.simulation_beginning_of_backwards_window,
                                                         self.simulation_end_of_future_window,
                                                         freq='1H')
        self.simulation_day_range = pd.date_range(str(self.simulation_beginning_of_backwards_window)[:10],
                                                  str(self.simulation_end_of_future_window)[:10], freq='1D')
        self.historical_data_length = (self.end_of_backwards_window - self.fitting_beginning_of_backwards_window).days

        """
        ramp factor based on time simulation is run (instead of a different ramp factor per simulation date)
        """

        time_since_beginning_of_spring = (self.simulation_beginning_of_future_window -
                                          pd.to_datetime('2021-03-01')).days
        if time_since_beginning_of_spring >= 0 and time_since_beginning_of_spring <= self.historical_data_length:
            self.ramp_fraction = (self.historical_data_length -
                                  time_since_beginning_of_spring) / self.historical_data_length
        else:
            self.ramp_fraction = -1

        # # DEBUG!!
        # self.ramp_fraction = 1

        self.logger.info(stringify('ramp fraction:', self.ramp_fraction))

        # processing card reader data
        self.card_reader_df, self.card_reader_buildings_list = self.transform_cardreader_json_dict_to_pandas()
        temp_card_reader_buildings_list = copy.deepcopy(self.card_reader_buildings_list)
        temp_card_reader_buildings_list.sort()
        self.logger.info(stringify('buildings in card reader:', temp_card_reader_buildings_list))

        self.all_buildings = list(set(self.card_reader_buildings_list + self.all_buildings))  # all_buildings tag

        self.arrival_counts_per_hour_pmf, self.all_arrivals_stats = self.make_input_arrival_counts_per_hour_pmf_dict()

        self.inflow_lists_building_dict = self.make_input_inflow_per_building_dict()

        self.logger.info(stringify('timeframe of data:', self.fitting_beginning_of_backwards_window,
                                   'to', self.end_of_backwards_window))
        self.logger.info(stringify('timeframe of simulation:', self.simulation_beginning_of_backwards_window,
                                   'to', self.simulation_end_of_future_window, '\n'))
        self.logger.info(stringify('historical_data_length:', self.historical_data_length))
        self.logger.info(stringify('ramp_fraction:', self.ramp_fraction))

    def make_holidays_list(self):
        """
        reads holiday dates and date-ranges
        """
        staff_holidays_raw = self.config['simulation_parameters']['staff_holidays']

        holidays_list = []
        for item in staff_holidays_raw:
            if isinstance(item, str):
                holidays_list.append(item)
            elif isinstance(item, list):
                if len(item) != 2:
                    error_message = 'holiday range can only have two values, start and end'
                    self.logger.exception(stringify(error_message))
                    raise ValueError(error_message)

                days = pd.date_range(item[0], item[1], freq='1D')
                days_str = [str(i)[:10] for i in days]
                holidays_list = holidays_list + days_str

        return holidays_list

    def read_schedule_injection(self):
        """
        read spring semester headcount to be injected later in arrivals
        """
        start = time.time()
        self.logger.info(stringify('[read_schedule_injection] reading spring schedule injection data'))
        in_person_class_headcount_json = self.config['simulation_parameters']['in_person_class_headcount']

        # with open(in_person_class_headcount_json) as fp:
        #     schedule_injection = json.load(fp)

        with gzip.open(in_person_class_headcount_json, 'r') as fin:
            schedule_injection = json.loads(fin.read().decode('utf-8'))

        schedule_injection_sanitizes = copy.deepcopy(schedule_injection)

        occupancy_building_keys = list(schedule_injection['in_person_class_headcount']['per_hour'].keys())
        inflow_building_keys = list(schedule_injection['in_person_class_inflow']['per_hour'].keys())
        outflow_building_keys = list(schedule_injection['in_person_class_outflow']['per_hour'].keys())

        # checking that building keys are the same
        if not(occupancy_building_keys == inflow_building_keys == outflow_building_keys):
            error_message = 'inflow, outflow and occupancy keys different in {}'.format(
                self.config['simulation_parameters']['inject_in_person_class_headcount'])
            self.logger.exception(stringify(error_message))
            raise ValueError(error_message)

        building_deltas = {}
        for blg in occupancy_building_keys:

            inflows = np.sum(
                list(schedule_injection['in_person_class_inflow']['per_hour'][blg].values())
            )
            outflows = np.sum(
                list(schedule_injection['in_person_class_outflow']['per_hour'][blg].values())
            )

            if inflows != outflows:
                error_message = 'inflow {} != outflow for blg {}!'.format(inflows, outflows, blg)
                self.logger.exception(stringify(error_message))
                raise ValueError(error_message)

            # checking that inflow outflow and occupancy have the same timestamps
            timestamps = list(schedule_injection['in_person_class_headcount']
                              ['per_hour'][occupancy_building_keys[0]].keys())
            blg_inflow_timestamps = list(schedule_injection['in_person_class_inflow']['per_hour'][blg].keys())
            blg_outflow_timestamps = list(schedule_injection['in_person_class_outflow']['per_hour'][blg].keys())
            if not (timestamps == blg_inflow_timestamps == blg_outflow_timestamps):
                error_message = 'timestamps between buildings or inflow/outflow/occupancy different'
                self.logger.exception(stringify(error_message))
                raise ValueError(error_message)

            # calculating delta due to taking sections of the spring schedule (which violates conservation due to edge
            # effects)
            if self.ramp_fraction != -1:
                beg_index = timestamps.index(str(self.simulation_beginning_of_future_window))
                end_index = timestamps.index(str(self.simulation_end_of_future_window)) + 1

                inflows = np.sum(
                    list(schedule_injection['in_person_class_inflow']['per_hour'][blg].values())[beg_index:end_index]
                )
                outflows = np.sum(
                    list(schedule_injection['in_person_class_outflow']['per_hour'][blg].values())[beg_index:end_index]
                )
                building_deltas[blg] = self.ramp_fraction*(inflows - outflows)

        self.logger.info(stringify('[read_schedule_injection] done in:', time.time() - start, '\n'))

        return schedule_injection, occupancy_building_keys, building_deltas

    def read_cardreader_json_to_dict(self):
        """
        reads json cardreader file and convert to a dictionary
        """

        start = time.time()

        self.logger.info(stringify('[read_cardreader_json_to_dict] reading card reader json'))
        """
		INGESTING FROM HISTORICAL_ANALYSIS BUCKET
		"""
        json_filepath = self.debug_cardreader_data
        metadata_json_filepath = self.debug_cardreader_metadata

        with open(metadata_json_filepath) as fp:
            metadata_dict = json.load(fp)

        end_of_backwards_window = pd.to_datetime(metadata_dict['historical_parameters']['interval_date_end']) - \
            timedelta(hours=1)

        earliest_time_from_card_reader_data = pd.to_datetime(metadata_dict['historical_parameters'][
            'interval_date_start'])

        # with open(json_filepath) as fp:
        #     card_reader_historical_dict = json.load(fp)

        with gzip.open(json_filepath, 'r') as fin:
            card_reader_historical_dict = json.loads(fin.read().decode('utf-8'))

        self.config['simulation_parameters']['card_reader_datafile'] = json_filepath
        self.config['simulation_parameters']['card_reader_metadata'] = metadata_dict

        ingestion_list_buildings = list(
            card_reader_historical_dict['card_reader'][self.cardreader_key_name]['per_hour'].keys())
        if 'NaN' in ingestion_list_buildings:
            ingestion_list_buildings.remove('NaN')
        if 'all' in ingestion_list_buildings:
            ingestion_list_buildings.remove('all')
        if 'null' in ingestion_list_buildings:
            ingestion_list_buildings.remove('null')
        if 'all_buildings' in ingestion_list_buildings:
            ingestion_list_buildings.remove('all_buildings')

        self.logger.info(stringify('[read_cardreader_json_to_dict] done in:', time.time() - start, '\n'))

        return card_reader_historical_dict, ingestion_list_buildings, end_of_backwards_window, earliest_time_from_card_reader_data

    def _thread_read_building_from_card_reader_dict(self, building):
        """
        prepares (select, transform) pandas data for each building
        """
        temp_building_dict = self.card_reader_historical_dict['card_reader'][self.cardreader_key_name]['per_hour'][
            building]
        # line below sums over MIT employee types (student, employee, other)
        temp_building_flattened_dict = {k: sum(v.values()) for k, v in temp_building_dict.items()}

        local_card_reader_df = pd.DataFrame.from_dict(temp_building_flattened_dict, orient='index', columns=['swipes'])

        local_card_reader_df['timestamp'] = local_card_reader_df.index
        local_card_reader_df['date'] = local_card_reader_df['timestamp'].astype(str).str[:10]
        local_card_reader_df['timestamp'] = pd.to_datetime(local_card_reader_df['timestamp'])

        # only taking data that's within out fittng window
        local_card_reader_df = local_card_reader_df[local_card_reader_df['timestamp']
                                                    >= self.fitting_beginning_of_backwards_window]
        local_card_reader_df = local_card_reader_df[local_card_reader_df['timestamp'] <= self.end_of_backwards_window]

        if local_card_reader_df.shape[0] > 0:
            local_card_reader_df['staff_holiday'] = False
            local_card_reader_df.loc[local_card_reader_df['date'].isin(self.holiday_list), 'staff_holiday'] = True
            local_card_reader_df['day_of_week'] = local_card_reader_df.apply(lambda x: x['timestamp'].weekday(), axis=1)
            local_card_reader_df['week_of_year'] = local_card_reader_df.apply(
                lambda x: int(x['timestamp'].strftime('%V')),
                axis=1)
            local_card_reader_df['hour_of_day'] = local_card_reader_df.apply(lambda x: x['timestamp'].hour, axis=1)
            local_card_reader_df['hour_of_week'] = local_card_reader_df['day_of_week'] * \
                24 + local_card_reader_df['hour_of_day']

            local_card_reader_df['location'] = building

            local_card_reader_df.reset_index(drop=True, inplace=True)

            return local_card_reader_df
        else:
            return pd.DataFrame()

    def transform_cardreader_json_dict_to_pandas(self):
        """
        outputs a dataframe of all card reader data (from dict)
        """

        start = time.time()
        self.logger.info(stringify('[transform_cardreader_json_dict_to_pandas] converting json to pandas dataframe'))

        # self.read_building_from_card_reader_dict('32')  # multiproc DEBUG
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        card_reader_df = pd.concat(pool.map(self._thread_read_building_from_card_reader_dict,
                                            self.ingestion_list_buildings))
        pool.close()
        pool.join()

        """
		example of card_reader_df.head()
		   swipes           timestamp  day_of_week  week_of_year  hour_of_day  \
		0       0 2020-09-29 00:00:00            1            40            0
		1       0 2020-09-29 01:00:00            1            40            1
		2       0 2020-09-29 02:00:00            1            40            2
		3       0 2020-09-29 03:00:00            1            40            3
		4       1 2020-09-29 04:00:00            1            40            4
		   hour_of_week location
		0            24      W35
		1            25      W35
		2            26      W35
		3            27      W35
		4            28      W35
		"""
        list_buildings = list(pd.unique(card_reader_df['location']))

        self.logger.info(stringify('[transform_cardreader_json_dict_to_pandas] done in:', time.time() - start, '\n'))

        return card_reader_df, list_buildings

    def make_input_arrival_counts_per_hour_pmf_dict(self):

        start = time.time()
        self.logger.info(stringify(
            '[make_input_arrival_counts_per_hour_pmf_dict] creating P(no. distinct people | building, day_of_week, staff_holiday)'))
        arrival_time_agg = self.card_reader_df[['swipes', 'hour_of_week', 'location', 'staff_holiday']]
        arrival_time_agg = arrival_time_agg.groupby(
            ['location', 'hour_of_week', 'staff_holiday']).agg(lambda tdf: tdf.tolist())
        arrival_time_agg.reset_index(inplace=True)
        arrival_time_agg.columns = ['location', 'hour_of_week', 'staff_holiday', 'possible_arrivals']
        arrival_counts_per_hour_pmf = {}

        all_normal_arrivals = []
        all_holiday_arrivals = []

        for building in tqdm(self.all_buildings):
            building_data = arrival_time_agg[arrival_time_agg['location'] == building]
            hour_of_week_dict = {}

            for hour_of_week in range(168):
                # missing data
                if hour_of_week not in pd.unique(building_data['hour_of_week']):
                    arrival_stats = {
                        'normal_day': {
                            'mean': np.nan,
                            'sd': np.nan
                        },
                        'holiday': {
                            'mean': np.nan,
                            'sd': np.nan
                        }
                    }
                else:
                    # normal_days
                    possible_values = building_data.loc[
                        (building_data['hour_of_week'] == hour_of_week)
                        &
                        (building_data['staff_holiday'] == False), 'possible_arrivals']

                    if possible_values.shape[0] > 0:
                        all_normal_arrivals = all_normal_arrivals + possible_values.values[0]
                        normal_day = {
                            'mean': np.mean(possible_values.values[0]),
                            'sd': np.std(possible_values.values[0]),
                        }
                    else:
                        normal_day = {
                            'mean': np.nan,
                            'sd': np.nan,
                        }

                    # holidays
                    possible_values = building_data.loc[
                        (building_data['hour_of_week'] == hour_of_week)
                        &
                        (building_data['staff_holiday'] == True), 'possible_arrivals']

                    if possible_values.shape[0] > 0:
                        all_holiday_arrivals = all_holiday_arrivals + possible_values.values[0]
                        holiday = {
                            'mean': np.mean(possible_values.values[0]),
                            'sd': np.std(possible_values.values[0]),
                        }
                    else:
                        holiday = {
                            'mean': np.nan,
                            'sd': np.nan,
                        }

                    arrival_stats = {
                        'normal_day': normal_day,
                        'holiday': holiday
                    }

                hour_of_week_dict[hour_of_week] = arrival_stats
            arrival_counts_per_hour_pmf[building] = hour_of_week_dict

        # mean and std over ALL time periods (to be used for times ie hour_of_week where we do not have data)
        all_arrivals_stats = {
            'normal_day': {
                'mean': np.mean(all_normal_arrivals),
                'sd': np.std(all_normal_arrivals)
            },
            'holiday': {
                'mean': np.mean(all_holiday_arrivals),
                'sd': np.std(all_holiday_arrivals)
            }
        }

        self.logger.info(stringify('[make_input_arrival_counts_per_hour_pmf_dict] done in:', time.time() - start, '\n'))
        return arrival_counts_per_hour_pmf, all_arrivals_stats

    def _thread_make_input_inflow_per_building_dict(self, building):

        try:
            this_building_data = self.card_reader_df[self.card_reader_df['location'] == building]

            # creating vector of inflow data
            inflow_data = []
            timestamps_present_for_this_building = this_building_data['timestamp'].to_list()

            for timestamp in self.simulation_full_time_window:
                # just missing values in the past
                if timestamp in timestamps_present_for_this_building and timestamp < self.simulation_beginning_of_future_window:
                    inflow_data.append(
                        this_building_data.loc[this_building_data['timestamp'] == timestamp, 'swipes'].values[0])
                # in cases when missing values peppered around sometimes
                elif timestamp not in timestamps_present_for_this_building and timestamp < \
                        self.simulation_beginning_of_future_window:
                    inflow_data.append(0)
                elif timestamp >= self.simulation_beginning_of_future_window:
                    inflow_data.append(None)
        except Exception as e:
            inflow_data = []
            for timestamp in self.simulation_full_time_window:
                inflow_data.append(None)

        if len(inflow_data) != len(self.simulation_full_time_window):
            error_message = 'missing data!'
            self.logger.exception(stringify(error_message))
            raise ValueError(error_message)

        return {'inflow_data': inflow_data,
                'building': building}

    def make_input_inflow_per_building_dict(self):
        '''
        this file takes the number of observed swipes for each building and
        converts it to a list to be used for pyro sampling down the line
        '''
        start = time.time()
        self.logger.info(stringify('[make_input_inflow_per_building_dict] creating lists of arrivals'))

        # test = self._thread_make_input_inflow_per_building_dict('32') #DEBUG

        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        inflow_per_building_list = pool.map(self._thread_make_input_inflow_per_building_dict, self.all_buildings)
        pool.close()
        pool.join()

        inflow_lists_building_dict = {}

        for item in inflow_per_building_list:
            inflow_lists_building_dict[item['building']] = item['inflow_data']

        self.logger.info(stringify('[make_input_inflow_per_building_dict] done in:', time.time() - start, '\n'))

        return inflow_lists_building_dict

    def _thread_make_assignment_buildings_count(self, building):
        this_building_assignments_pmf = {}

        for t in self.building_assignments_raw[building].keys():
            timestamp = pd.to_datetime(t)
            this_blg_this_time_dict_not_flattened = self.building_assignments_raw[building][t]
            # line below sums over MIT employee types (student, employee, other)
            this_blg_this_time_dict = {k: sum(v.values())
                                       for k, v in this_blg_this_time_dict_not_flattened.items()}

            if 'student_buildings' in this_blg_this_time_dict:
                student_building_count = this_blg_this_time_dict['student_buildings']
                # remove key as we're going to unpack/add it below
                del this_blg_this_time_dict['student_buildings']
                for student_blg in self.student_buildings:
                    if student_blg in this_blg_this_time_dict:
                        this_blg_this_time_dict[student_blg] = this_blg_this_time_dict[student_blg] + \
                            student_building_count
                    else:
                        this_blg_this_time_dict[student_blg] = student_building_count

            if 'all_buildings' in this_blg_this_time_dict:
                all_buildings_count = this_blg_this_time_dict['all_buildings']
                # remove key as we're going to unpack/add it below
                del this_blg_this_time_dict['all_buildings']
                for a_blg in self.all_buildings:
                    if a_blg in this_blg_this_time_dict:
                        this_blg_this_time_dict[a_blg] = this_blg_this_time_dict[a_blg] + all_buildings_count
                    else:
                        this_blg_this_time_dict[a_blg] = all_buildings_count

            t = str(pd.to_datetime(t))
            this_building_assignments_pmf[t] = this_blg_this_time_dict

        this_building_assignments_df = pd.DataFrame.from_dict(this_building_assignments_pmf,
                                                              orient='index')

        # only taking data that's within out fittng window
        this_building_assignments_df['timestamp'] = pd.to_datetime(this_building_assignments_df.index)

        this_building_assignments_df = this_building_assignments_df[this_building_assignments_df['timestamp']
                                                                    >= self.fitting_beginning_of_backwards_window]

        if this_building_assignments_df.shape[0] == 0:
            """
            sometimes building assignments are empty because of fitting window
            or the data we get is empty e.g.:
            "42": {
                    "2020-10-31 8:00:00": {},
                    "2020-12-23 9:00:00": {},
                    "2020-12-11 9:00:00": {},
                    "2021-01-21 9:00:00": {},
                    ...
            """
            this_building_dict = {}

        else:

            this_building_assignments_df['day_of_week'] = this_building_assignments_df.apply(
                lambda x: x['timestamp'].weekday(), axis=1)
            this_building_assignments_df['hour_of_day'] = this_building_assignments_df.apply(
                lambda x: x['timestamp'].hour, axis=1)
            this_building_assignments_df['hour_of_week'] = this_building_assignments_df['day_of_week'] * \
                24 + this_building_assignments_df['hour_of_day']

            this_building_assignments_df = this_building_assignments_df.drop(
                columns=['timestamp', 'hour_of_day', 'day_of_week'])

            columns = this_building_assignments_df.columns
            columns = columns[:-1]
            columns = columns.insert(0, 'hour_of_week')

            this_building_assignments_agg_df = this_building_assignments_df.groupby(['hour_of_week']).agg(['sum'])
            this_building_assignments_agg_df.reset_index(inplace=True)
            this_building_assignments_agg_df.columns = columns

            this_building_assignments_agg_long = pd.melt(this_building_assignments_agg_df, id_vars=['hour_of_week'])
            this_building_assignments_agg_long.columns = ['hour_of_week', 'assigned_building', 'sum_counts']
            this_building_assignments_agg_long['probability'] = 0

            this_building_dict = {}
            hour_list = pd.unique(this_building_assignments_df['hour_of_week'])
            hour_list.sort()

            # this is where we take sum counts of assignments by hour of the week and calculate building-wise
            # probability, then convert to a dictionary
            for h in hour_list:
                total = np.nansum(
                    this_building_assignments_agg_long[this_building_assignments_agg_long['hour_of_week'] == h]
                    ['sum_counts']
                )

                this_building_assignments_agg_long.loc[
                    this_building_assignments_agg_long['hour_of_week'] == h, 'probability'] = \
                    this_building_assignments_agg_long[this_building_assignments_agg_long['hour_of_week']
                                                       == h]['sum_counts'] / total

                temp = this_building_assignments_agg_long[this_building_assignments_agg_long['hour_of_week'] == h]
                temp = temp[['assigned_building', 'probability']]
                temp.index = temp['assigned_building']
                temp_dict = temp.to_dict()
                this_building_dict[h] = temp_dict['probability']

        return {'this_building_assignments_pmf': this_building_dict,
                'building': building}

    def make_assignment_buildings_count(self):
        """
        reads the assignment_buildings_count data (i.e. assignment buildings of
        people who swiped in a certain building at a certain time
        """

        start = time.time()
        self.logger.info(stringify('[make_assignment_buildings_count] creating assignment building data'))

        self.building_assignments_raw = self.card_reader_historical_dict['card_reader']['assignment_buildings_count'][
            'per_hour']

        building_to_iterate_over = list(self.building_assignments_raw.keys())

        # self._thread_make_assignment_buildings_count('32') #DEBUG

        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        this_building_assignments_pmf_list = pool.map(
            self._thread_make_assignment_buildings_count, building_to_iterate_over)
        pool.close()
        pool.join()

        all_buildings_assignments_pmf = {}

        for item in this_building_assignments_pmf_list:
            all_buildings_assignments_pmf[item['building']] = item['this_building_assignments_pmf']

        self.logger.info(stringify('[make_assignment_buildings_count] done in:', time.time() - start, '\n'))
        return all_buildings_assignments_pmf

    def make_durations_pmf_dict(self):

        start = time.time()
        self.logger.info(stringify('[make_durations_pmf_dict] creating duration wifi pmf'))
        """
		Reading and formatting schedule of arrival-depature time so we end up with dicts of duration time
		"""

        duration_json_filepath = self.config['simulation_parameters']['data_parameters']['duration_pmf']
        with open(duration_json_filepath, 'r') as fp:
            temp_schedule_cond_person_pmf = json.load(fp)

        self.config['simulation_parameters']['duration_datafile'] = duration_json_filepath

        duration_dict = temp_schedule_cond_person_pmf['hourly_durations_probs']['per_hour']

        all_durations = []
        final_duration_dict = {}
        for building in duration_dict.keys():
            building_duration = duration_dict[building]
            building_duration_dict = {}
            if len(building_duration.keys()) != 168:
                error_message = 'building should have 168 hour_of_week, only has {}'.format(
                    len(building_duration.keys()))
                self.logger.exception(stringify(error_message))
                raise ValueError(error_message)

            for hour_of_week in building_duration.keys():
                duration_prob = building_duration[hour_of_week]['stay_durations']
                durations = [int(d) for d in list(duration_prob.keys())]
                all_durations = all_durations + durations
                if sum(list(duration_prob.values())) > 0:
                    probs = [p / sum(list(duration_prob.values())) for p in list(duration_prob.values())]
                else:
                    probs = [0 for p in list(duration_prob.values())]
                new_duration_prob = {}
                for i in range(len(durations)):
                    new_duration_prob[durations[i]] = probs[i]

                building_duration_dict[int(hour_of_week)] = new_duration_prob
            final_duration_dict[building] = building_duration_dict

        self.logger.info(stringify('[make_durations_pmf_dict] done in:', time.time() - start, '\n'))
        return final_duration_dict, np.mean(all_durations), np.std(all_durations)

    def make_buildings_transition_pmf(self):
        """
        reads the transition json file and outputs a dictionary that looks like this:
                {
                "14": {
                        "1": {
                                "Outside": 0.20435574961437022,
                                "4": 0.11546250684181719,
                                "6": 0.02162014230979748,
                                "3": 0.005747126436781617,
                                "7": 0.005747126436781617,
                                ...}
                        ...}
                }
        """
        start = time.time()
        self.logger.info(stringify('[make_buildings_transition_pmf] creating transitions wifi pmf'))

        transition_json_filepath = self.config['simulation_parameters']['data_parameters']['transition_probs']
        with open(transition_json_filepath, 'r') as fp:
            temp_transition_dict = json.load(fp)

        self.config['simulation_parameters']['transition_datafile'] = transition_json_filepath

        # we don't want to model people from going outside then back in (for now)
        if 'Outside' in temp_transition_dict['hourly_transition_probs']['per_hour'].keys():
            del temp_transition_dict['hourly_transition_probs']['per_hour']['Outside']

        transition_dict = temp_transition_dict['hourly_transition_probs']['per_hour']
        final_transition_dict = {}
        for building in transition_dict.keys():
            building_transition = transition_dict[building]
            building_transition_dict = {}
            hours = [int(d) for d in list(building_transition.keys())]
            hours.sort()
            for hour_of_week in hours:
                transition_prob = building_transition[str(hour_of_week)]
                transitions = [d for d in list(transition_prob.keys())]
                if sum(list(transition_prob.values())) > 0:
                    probs = [p / sum(list(transition_prob.values())) for p in list(transition_prob.values())]
                    if sum(probs) < 0.999:
                        error_message = 'prob should sum to 1.0 for {} in hour {}'.format(building, hour_of_week)
                        self.logger.exception(stringify(error_message))
                        raise ValueError(error_message)
                else:
                    probs = [0 for p in list(transition_prob.values())]
                new_transition_prob = {}
                for i in range(len(transitions)):
                    new_transition_prob[transitions[i]] = probs[i]

                building_transition_dict[hour_of_week] = new_transition_prob
            final_transition_dict[building] = building_transition_dict

        self.logger.info(stringify('[make_buildings_transition_pmf] done in:', time.time() - start, '\n'))
        return final_transition_dict

    def make_output_past_future_dict(self):
        """
        runs sampling via parallel replications for inflow and occupancy
        """
        # self.all_samples_list = [self.replicate_simulation(1)]  # multiproc DEBUG

        start = time.time()
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        self.logger.info(stringify('[make_output_past_future_dict] they see me sampling...'))
        self.all_samples_list = pool.map(self.replicate_simulation, range(self.N_REPLICATIONS))
        pool.close()
        pool.join()
        self.logger.info(
            stringify('[make_output_past_future_dict] All sampling finished in', time.time() - start, '\n'))

    def replicate_simulation(self, replication: int):
        """
        exectutes the inflow and occupancy models
        """

        random.seed(replication)
        np.random.seed(replication)

        inflow_start = time.time()
        before_transitions_arrivals_dict = self.inflow_model()

        occupancy_start = time.time()
        occupancy_dict, outflow_dict, updated_adjacency_matrix, arrivals_dict = self.occupancy_model(
            before_transitions_arrivals_dict)

        self.logger.info(stringify('[replicate_simulation {}]'.format(replication),
                                   'inflow:'.format(str(replication)), occupancy_start - inflow_start,
                                   'occupancy: ', time.time() - occupancy_start,
                                   '\n'))

        return {
            'arrivals_dict': arrivals_dict,
            'occupancy_dict': occupancy_dict,
            'outflow_dict': outflow_dict,
            'adjacency_matrix': updated_adjacency_matrix
        }

    def empirical_situational_inflow_model(self):
        """
        function that actually does the inflow/arrival sampling using pyro
        """
        arrivals_dict = {}

        for building in self.all_buildings:
            arrivals = np.asarray([0.0 for _ in range(len(self.simulation_full_time_window))])

            observed_arrivals = self.inflow_lists_building_dict[building]
            for t, timestamp in enumerate(self.simulation_full_time_window):
                hour_of_week = convert_timestamp_to_hour_of_week(timestamp)

                if str(timestamp)[:10] in self.holiday_list:
                    if np.isnan(self.arrival_counts_per_hour_pmf[building][hour_of_week]['holiday']['mean']):
                        # if we really have no holiday data
                        if np.isnan(self.all_arrivals_stats['holiday']['mean']):
                            # if we even don't have any data
                            if np.isnan(self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['mean']):
                                mean_arrival = self.all_arrivals_stats['normal_day']['mean']
                                std_arrival = self.all_arrivals_stats['normal_day']['sd']
                            else:
                                mean_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['mean']
                                std_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['sd']
                        else:
                            mean_arrival = self.all_arrivals_stats['holiday']['mean']
                            std_arrival = self.all_arrivals_stats['holiday']['sd']
                    else:
                        mean_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['holiday']['mean']
                        std_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['holiday']['sd']
                else:
                    if np.isnan(self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['mean']):
                        mean_arrival = self.all_arrivals_stats['normal_day']['mean']
                        std_arrival = self.all_arrivals_stats['normal_day']['sd']
                    else:
                        mean_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['mean']
                        std_arrival = self.arrival_counts_per_hour_pmf[building][hour_of_week]['normal_day']['sd']

                if observed_arrivals[t] is None:
                    n_arrivals_this_hour = int(np.random.normal(mean_arrival, std_arrival))
                else:
                    n_arrivals_this_hour = observed_arrivals[t]

                # we cannot have negative arrivals (this essentially implements a +half-Normal)
                if n_arrivals_this_hour < 0:
                    n_arrivals_this_hour = 0
                if n_arrivals_this_hour == None:
                    error_message = 'n_arrivals_this_hour is None'
                    self.logger.exception(stringify(error_message))
                    raise ValueError(error_message)

                arrivals[t] = n_arrivals_this_hour

            arrivals_dict[building] = arrivals

        return arrivals_dict

    # profiling decorator for later use
    # @profile
    def empirical_situational_occupancy_model(self, before_transitions_arrivals_dict: dict):
        """
        function that actually does the occupancy sampling using pyro
        """

        num_entities = len(self.all_buildings)
        adjacency_matrix = np.zeros((num_entities, num_entities))

        occupancy_dict = {}
        outflow_dict = {}

        updated_arrivals_dict = copy.deepcopy(before_transitions_arrivals_dict)

        # creating empty dictionary data
        injected_delta = {}
        for building in self.all_buildings:
            occupancy_dict[building] = np.asarray([0.0 for _ in range(len(self.simulation_full_time_window))])
            outflow_dict[building] = np.asarray([0.0 for _ in range(len(self.simulation_full_time_window))])
            injected_delta[building] = 0

        if self.limited_buiding_list == True:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = self.card_reader_buildings_list

        for starting_building in tqdm(building_to_iterate_over):

            for t, timestamp in enumerate(self.simulation_full_time_window):
                hour_of_week = convert_timestamp_to_hour_of_week(timestamp)

                # sometimes we don't have duration data for some building-time combination
                if starting_building in self.duration_pmf_dict:
                    duration_present_for_this_building = True
                    duration_dict = self.duration_pmf_dict[starting_building][hour_of_week]
                    # getting duration values and probs
                    possible_durations = list(duration_dict.keys())
                    prob_duration = list(self.duration_pmf_dict[starting_building][hour_of_week].values())
                else:
                    duration_present_for_this_building = False
                    # duration = self.mean_duration

                    # sometimes we don't have assignment data for some building-time combination
                try:
                    assigned_buildings_pmf = self.all_buildings_assignments_pmf[starting_building][hour_of_week]
                except KeyError as e:
                    assigned_buildings_pmf = {}

                # for this hour_of_week t, for this building, we obtain a number of arrivals from arrivals_dict[
                # building]. for each swipe (arrival), we then sample a duration and update the occupancy downstream
                for a in range(int(before_transitions_arrivals_dict[starting_building][t])):

                    if duration_present_for_this_building:
                        if np.sum(list(prob_duration)) == 0:
                            stay_duration = np.random.normal(self.mean_duration, self.std_duration)
                        else:
                            stay_duration = random.choices(population=possible_durations,
                                                           weights=prob_duration,
                                                           k=1)[0]
                    else:
                        stay_duration = np.random.normal(self.mean_duration, self.std_duration)

                    stay_duration = int(stay_duration)
                    if stay_duration < 0:
                        stay_duration = 0

                    # sometimes we have an empty assignment pmf for some building-time combination
                    if not assigned_buildings_pmf.keys():
                        destination_building = starting_building
                    else:
                        destination_building, path_through_buildings = self.building_transition_function(
                            starting_building,
                            assigned_buildings_pmf,
                            hour_of_week,
                            timestamp)
                    # updating adjacency matrix
                    source_building_index = self.all_buildings.index(starting_building)
                    dest_building_index = self.all_buildings.index(destination_building)
                    adjacency_matrix[source_building_index][dest_building_index] = \
                        adjacency_matrix[source_building_index][dest_building_index] + 1

                    # getting end time of arrival
                    if t + stay_duration + 1 < outflow_dict[starting_building].shape[0]:
                        end_time_of_stay = t + stay_duration + 1
                    else:
                        # clip time to the max possible value
                        end_time_of_stay = outflow_dict[starting_building].shape[0] - 1

                    # if there was a transition from starting_building to destination_building
                    if destination_building != starting_building:

                        # person leaves starting_building immediately
                        outflow_dict[starting_building][t] += 1

                        # # person goes into destination_building then leaves that building at end_time_of_stay
                        updated_arrivals_dict[destination_building][t] += 1
                        outflow_dict[destination_building][end_time_of_stay] += 1

                        # occupancy is updated
                        occupancy_dict[destination_building][t: end_time_of_stay] += 1

                        # note: path_through_buildings never contains destination_building
                        for transition_blg in path_through_buildings:
                            if transition_blg != starting_building:
                                try:
                                    updated_arrivals_dict[transition_blg][t] = updated_arrivals_dict[transition_blg][
                                        t] + 1
                                    outflow_dict[transition_blg][t] = outflow_dict[transition_blg][t] + 1

                                # if we end up with buildings outside the set of buildings we considered
                                except KeyError as e:
                                    # in case needed for debug
                                    # print('path_through_buildings error', e)
                                    pass

                    # there was no transition
                    else:
                        # # corresponding outflow
                        outflow_dict[starting_building][end_time_of_stay] += 1

                        # occupancy is updated
                        occupancy_dict[starting_building][t: end_time_of_stay] += 1

                if self.inject_in_person_class_headcount and \
                        starting_building in self.schedule_injection['in_person_class_headcount']['per_hour'] and \
                        self.ramp_fraction >= 0 and \
                        timestamp >= self.simulation_beginning_of_future_window and \
                        str(timestamp) in self.schedule_injection['in_person_class_headcount']['per_hour'][starting_building]:

                    delta_occupancy = self.ramp_fraction * \
                        self.schedule_injection['in_person_class_headcount']['per_hour'][starting_building][str(
                            timestamp)]
                    occupancy_dict[starting_building][t] = float(occupancy_dict[starting_building][t]) + \
                        delta_occupancy

                    delta_inflow = self.ramp_fraction * \
                        self.schedule_injection['in_person_class_inflow']['per_hour'][starting_building][str(timestamp)]
                    updated_arrivals_dict[starting_building][t] = float(updated_arrivals_dict[starting_building][t]) + \
                        delta_inflow

                    delta_outflow = self.ramp_fraction * \
                        self.schedule_injection['in_person_class_outflow']['per_hour'][starting_building][str(
                            timestamp)]
                    outflow_dict[starting_building][t] = float(outflow_dict[starting_building][t]) + \
                        delta_outflow

        return occupancy_dict, outflow_dict, adjacency_matrix, updated_arrivals_dict  # , transitions_dict

    def unity_transition_swipe_to_assigned_building(self, starting_building, assigned_buildings_pmf, hour_of_week,
                                                    timestamp):
        return starting_building, []

    def run_transition_swipe_to_assigned_building(self, starting_building, assigned_buildings_pmf, hour_of_week,
                                                  timestamp):
        """
        given a starting_building (where people tap their access card), we are trying to
        figure out which building are they actually trying to get to so we can update the occupancy of that building.
        We do so by repeatedly applying a transition_matrix until the output building
        is one of the assigned buildings
        """

        # hopefully no building at MIT is named liked this in the future (still better than dealing with NaN)
        destination_building = 'fail'

        for assi in range(self.num_trials_assignments):

            # if a previous look already found a starting building
            if destination_building != starting_building and destination_building != 'fail':
                break

            current_building = starting_building

            # (re)setting the path_through_buildings to empty. Not adding first since swipe blg already has arrival
            path_through_buildings = []
            path_through_buildings.append(starting_building)

            work_building_trial = random.choices(
                population=list(assigned_buildings_pmf.keys()),
                weights=list(assigned_buildings_pmf.values()),
                k=1
            )[0]

            if work_building_trial == starting_building:
                destination_building = work_building_trial
                break

            for _ in range(self.num_trials_transitions):
                try:
                    current_transition_matrix = self.all_buildings_transition_pmf[current_building][hour_of_week]
                except Exception as e:
                    break

                trial_current_building = random.choices(
                    population=list(current_transition_matrix.keys()),
                    weights=list(current_transition_matrix.values()),
                    k=1
                )[0]
                if np.sum(list(current_transition_matrix.values())) < 0.99 or np.sum(list(
                        current_transition_matrix.values())) > 1.01:
                    error_message = 'doesnt sum {}'.format(np.sum(list(current_transition_matrix.values())))
                    self.logger.exception(stringify(error_message))
                    raise ValueError(error_message)

                # we don't want to model people from going outside then back in (for now)
                if trial_current_building != 'Outside':
                    current_building = trial_current_building
                    path_through_buildings.append(current_building)
                else:
                    continue

                if work_building_trial == current_building:
                    destination_building = current_building
                    break

        # the algo above did not end up on an assigned building
        if destination_building == 'fail':
            destination_building = starting_building
            path_through_buildings = []

        # set used to removed repeated buildings from transitions
        path_through_buildings = list(set(path_through_buildings))

        # remove destination_building from path_through_buildings to avoice destination_building double counting
        # buiding statistics
        if destination_building in path_through_buildings:
            path_through_buildings.remove(destination_building)

        return destination_building, path_through_buildings

    def make_output_json(self):
        """
        saving results as per JSON requirement
        """
        self.logger.info(stringify('[make_output_json] creating final dict output'))

        start = time.time()

        hourly_final_inflow_dict = {}
        hourly_final_outflow_dict = {}
        hourly_final_occupancy_dict = {}

        day_final_inflow_dict = {}
        day_final_occupancy_dict = {}

        """
		calculating:
			hourly inflow
			hourly occupancy

		"""

        if self.limited_buiding_list:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = self.card_reader_buildings_list

        for building in tqdm(building_to_iterate_over):
            inflow_replication_times_time_array = np.asarray([self.all_samples_list[r]['arrivals_dict'][building]
                                                              for r in range(len(self.all_samples_list))])

            outflow_replication_times_time_array = np.asarray([self.all_samples_list[r]['outflow_dict'][building]
                                                               for r in range(len(self.all_samples_list))])

            occupancy_replication_times_time_array = np.asarray([self.all_samples_list[r]['occupancy_dict'][building]
                                                                 for r in range(len(self.all_samples_list))])

            # really important check to make sure people are conserved
            # we can't just compare and have to use 0.1 episilon due to numpy's float low precision
            # otherwise we get it off by tiny values e.g. 2.2737367544323206e-13
            if building in self.building_deltas:
                if (np.sum(inflow_replication_times_time_array) - np.sum(outflow_replication_times_time_array)) - \
                        (float(self.N_REPLICATIONS)*self.building_deltas[building]) > 0.1:
                    error_message = 'building {}: inflow {} != outflow {}. uh oh, violation of ' \
                                    'conservation of people'.format(
                                        building,
                                        np.sum(inflow_replication_times_time_array),
                                        np.sum(outflow_replication_times_time_array)
                                    )
                    self.logger.exception(stringify(error_message))
                    raise ValueError(error_message)
                else:
                    if (np.sum(inflow_replication_times_time_array) - np.sum(outflow_replication_times_time_array)) > 0.1:
                        error_message = 'building {}: inflow {} != outflow {}. uh oh, violation of ' \
                                        'conservation of people'.format(
                                            building,
                                            np.sum(inflow_replication_times_time_array),
                                            np.sum(outflow_replication_times_time_array)
                                        )
                        self.logger.exception(stringify(error_message))
                        raise ValueError(error_message)

            """
			example of what above array looks like:
				of shape [N_REPLICATIONS (row) x num timestamps (column ie len of list)]
			array([[ 0, 15,  4, ..., 22, 22, 36],
			   [ 0, 15,  4, ..., 16, 37, 36],
			   [ 0, 15,  4, ..., 16, 37, 25],
			   ...,
			   [ 0, 15,  4, ..., 16, 28, 25],
			   [ 0, 15,  4, ..., 22, 37, 36],
			   [ 0, 15,  4, ..., 22, 37, 36]])
			"""

            # inflow mean and std
            mean_inflow_building_data = np.asarray([self.all_samples_list[r]['arrivals_dict'][building]
                                                    for r in range(len(self.all_samples_list))]).mean(axis=0)
            std_inflow_building_data = np.asarray([self.all_samples_list[r]['arrivals_dict'][building]
                                                   for r in range(len(self.all_samples_list))]).std(axis=0)

            # occupancy mean and std
            mean_occupancy_building_data = np.asarray(
                [self.all_samples_list[r]['occupancy_dict'][building] for r in
                 range(len(self.all_samples_list))]).mean(
                axis=0)
            std_occupancy_building_data = np.asarray(
                [self.all_samples_list[r]['occupancy_dict'][building] for r in
                 range(len(self.all_samples_list))]).std(
                axis=0)

            # outflow
            mean_outflow_building_data = np.asarray([self.all_samples_list[r]['arrivals_dict'][building]
                                                     for r in range(len(self.all_samples_list))]).mean(axis=0)
            std_outflow_building_data = np.asarray([self.all_samples_list[r]['arrivals_dict'][building]
                                                    for r in range(len(self.all_samples_list))]).std(axis=0)

            hourly_building_inflow = {}
            hourly_building_outflow = {}
            hourly_building_occupancy = {}

            day_building_inflow = {}
            day_building_occupancy = {}

            for t, timestamp in enumerate(self.simulation_full_time_window):

                hourly_inflow_stats = {
                    'mean': sig_fig_formatted(mean_inflow_building_data[t], num_digits=3),
                    'std': sig_fig_formatted(std_inflow_building_data[t], num_digits=3),
                }

                hourly_inflows_this_timestamp = inflow_replication_times_time_array[:, t]
                # percentile over replication values
                hourly_inflow_percentiles_vals = np.percentile(hourly_inflows_this_timestamp, self.percentiles_list,
                                                               interpolation='nearest')

                for p, val in enumerate(self.percentiles_list):
                    if hourly_inflow_percentiles_vals[p] is not None:
                        hourly_inflow_stats[str(val) + ' percentile'] = sig_fig_formatted(
                            hourly_inflow_percentiles_vals[p],
                            num_digits=3)
                    else:
                        hourly_inflow_stats[str(val) + ' percentile'] = 0

                hourly_building_inflow[str(timestamp)] = hourly_inflow_stats

                # occupancy
                hourly_occupancy_stats = {
                    'mean': sig_fig_formatted(mean_occupancy_building_data[t], num_digits=3),
                    'std': sig_fig_formatted(std_occupancy_building_data[t], num_digits=3),
                }
                hourly_building_occupancy[str(timestamp)] = hourly_occupancy_stats

                hourly_occupancy_this_timestamp = occupancy_replication_times_time_array[:, t]
                hourly_occupancy_percentiles_vals = np.percentile(hourly_occupancy_this_timestamp,
                                                                  self.percentiles_list,
                                                                  interpolation='nearest')

                for p, val in enumerate(self.percentiles_list):
                    if hourly_occupancy_percentiles_vals[p] is not None:
                        hourly_occupancy_stats[str(val) + ' percentile'] = sig_fig_formatted(
                            hourly_occupancy_percentiles_vals[p],
                            num_digits=3)
                    else:
                        hourly_occupancy_stats[str(val) + ' percentile'] = 0

                # outflow
                hourly_outflow_stats = {
                    'mean': sig_fig_formatted(mean_outflow_building_data[t], num_digits=3),
                    'std': sig_fig_formatted(std_outflow_building_data[t], num_digits=3),
                }

                hourly_outflows_this_timestamp = outflow_replication_times_time_array[:, t]
                hourly_outflow_percentiles_vals = np.percentile(hourly_outflows_this_timestamp, self.percentiles_list,
                                                                interpolation='nearest')

                for p, val in enumerate(self.percentiles_list):
                    if hourly_outflow_percentiles_vals[p] is not None:
                        hourly_outflow_stats[str(val) + ' percentile'] = sig_fig_formatted(
                            hourly_outflow_percentiles_vals[p],
                            num_digits=3)
                    else:
                        hourly_outflow_stats[str(val) + ' percentile'] = 0

                hourly_building_outflow[str(timestamp)] = hourly_outflow_stats

            hourly_final_inflow_dict[building] = hourly_building_inflow
            hourly_final_outflow_dict[building] = hourly_building_outflow
            hourly_final_occupancy_dict[building] = hourly_building_occupancy

            """
			calculating:
				peak_daily_occupancy
				peak_daily_inflow

			"""

            daily_building_inflow = {}
            daily_building_occupancy = {}
            for d, day in enumerate(self.simulation_day_range):
                if day != self.simulation_day_range[-1]:
                    index_times_this_day = np.argwhere(
                        (self.simulation_full_time_window > self.simulation_day_range[d])
                        &
                        (self.simulation_full_time_window < self.simulation_day_range[d + 1])
                    ).flatten().tolist()
                else:  # last day only
                    index_times_this_day = np.argwhere(
                        self.simulation_full_time_window > self.simulation_day_range[d]
                    ).flatten().tolist()

                """
				index_times_this_day is a list, containing e.g. [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
				the length being the number of replications
				"""
                inflow_day_max_list = []
                occupancy_day_max_list = []

                for replication in range(self.N_REPLICATIONS):
                    # inflow
                    inflow_replication_only_array = inflow_replication_times_time_array[replication]
                    inflow_day_slice = inflow_replication_only_array[index_times_this_day[0]:index_times_this_day[-1]]
                    # this gives the max (peak) value over all hours for this day for this replication
                    inflow_day_max_list.append(np.max(inflow_day_slice))

                    # occupancy
                    occupancy_replication_only_array = occupancy_replication_times_time_array[replication]
                    occupancy_day_slice = occupancy_replication_only_array[index_times_this_day[0]:index_times_this_day[-1]]
                    occupancy_day_max_list.append(np.max(occupancy_day_slice))

                # inflow
                day_inflow_stat = {
                    'mean': sig_fig_formatted(np.mean(inflow_day_max_list), num_digits=3),
                    'std': sig_fig_formatted(np.std(inflow_day_max_list), num_digits=3)
                }

                day_inflow_percentiles_vals = np.percentile(inflow_day_max_list, self.percentiles_list,
                                                            interpolation='nearest')

                for p, val in enumerate(self.percentiles_list):
                    if day_inflow_percentiles_vals[p] is not None:
                        day_inflow_stat[str(val) + ' percentile'] = sig_fig_formatted(day_inflow_percentiles_vals[p],
                                                                                      num_digits=3)
                    else:
                        day_inflow_stat[str(val) + ' percentile'] = 0

                daily_building_inflow[str(day)] = day_inflow_stat

                # occupancy
                occupancy_day_stat = {
                    'mean': sig_fig_formatted(np.mean(occupancy_day_max_list), num_digits=3),
                    'std': sig_fig_formatted(np.std(occupancy_day_max_list), num_digits=3)
                }

                day_occupancy_percentiles_vals = np.percentile(occupancy_day_max_list, self.percentiles_list,
                                                               interpolation='nearest')

                for p, val in enumerate(self.percentiles_list):
                    if day_occupancy_percentiles_vals[p] is not None:
                        occupancy_day_stat[str(val) + ' percentile'] = sig_fig_formatted(
                            day_occupancy_percentiles_vals[p],
                            num_digits=3)
                    else:
                        occupancy_day_stat[str(val) + ' percentile'] = 0

                daily_building_occupancy[str(day)] = occupancy_day_stat

            day_final_inflow_dict[building] = daily_building_inflow
            day_final_occupancy_dict[building] = daily_building_occupancy

        output_json = {
            'simulation': {
                'inflow': {
                    'per_hour': sort_dict(hourly_final_inflow_dict)
                },
                'occupancy': {
                    'per_hour': sort_dict(hourly_final_occupancy_dict)
                },
                'outflow': {
                    'per_hour': sort_dict(hourly_final_outflow_dict)
                },
                'peak_daily_inflow': {
                    'per_day': sort_dict(day_final_inflow_dict)
                },
                'peak_daily_occupancy': {
                    'per_day': sort_dict(day_final_occupancy_dict)
                }
            }
        }

        self.logger.info(stringify('[make_output_json]', time.time() - start, '\n'))

        return output_json

    def create_ground_truth_comparison(self):
        """
        comparing to ground truth
        """

        self.logger.info(stringify('creating final dict output'))

        start = time.time()

        hourly_final_inflow_dict = {}
        hourly_final_outflow_dict = {}
        hourly_final_occupancy_dict = {}

        day_final_inflow_dict = {}
        day_final_occupancy_dict = {}

        """
		calculating:
			hourly inflow
			hourly occupancy

		"""
        probability_of_data = {}

        for building in tqdm(self.card_reader_buildings_list):

            if building not in self.buildings_under_scrutiny:
                continue

            inflow_ground_truth = self.ground_truth_dict[building][
                list(self.ground_truth_dict[building].keys())[0]
            ]['inflow']

            outflow_ground_truth = self.ground_truth_dict[building][
                list(self.ground_truth_dict[building].keys())[0]
            ]['outflow']

            inflow_replication_times_time_array = np.asarray([self.all_samples_list[r]['arrivals_dict'][
                building]
                for r in range(len(self.all_samples_list))])
            occupancy_replication_times_time_array = np.asarray(
                [self.all_samples_list[r]['occupancy_dict'][building]
                 for r in range(len(self.all_samples_list))])

            outflow_replication_times_time_array = np.asarray(
                [self.all_samples_list[r]['outflow_dict'][building]
                 for r in range(len(self.all_samples_list))])

            """
			example of what above array looks like:
				of shape [N_REPLICATIONS x num timestamps]
			array([[ 0, 15,  4, ..., 22, 22, 36],
			   [ 0, 15,  4, ..., 16, 37, 36],
			   [ 0, 15,  4, ..., 16, 37, 25],
			   ...,
			   [ 0, 15,  4, ..., 16, 28, 25],
			   [ 0, 15,  4, ..., 22, 37, 36],
			   [ 0, 15,  4, ..., 22, 37, 36]])
			"""

            inflow_list_of_replications = []
            outflow_list_of_replications = []
            occupancy_list_of_replications = []

            for t, timestamp in enumerate(self.full_time_window):

                if timestamp >= self.ground_truth_data_start and timestamp <= self.ground_truth_data_end:
                    # inflow
                    hourly_inflows_this_timestamp = inflow_replication_times_time_array[:, t]
                    inflow_list_of_replications.append(hourly_inflows_this_timestamp)

                    # outflow
                    hourly_outflows_this_timestamp = outflow_replication_times_time_array[:, t]
                    outflow_list_of_replications.append(hourly_outflows_this_timestamp)

                    # occupancy
                    hourly_occupancy_this_timestamp = occupancy_replication_times_time_array[:, t]
                    occupancy_list_of_replications.append(hourly_occupancy_this_timestamp)

            inflow_replications_this_day = sum(inflow_list_of_replications)
            outflow_replications_this_day = sum(outflow_list_of_replications)
            occupancy_replications_this_day = sum(occupancy_list_of_replications)

            # occupancy

            if not os.path.exists(self.plot_location):
                os.makedirs(self.plot_location)

            if not os.path.exists('{}/{}'.format(self.plot_location, 'histogram')):
                os.makedirs('{}/{}'.format(self.plot_location, 'histogram'))

            """
			INFLOW
			"""

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.hist(inflow_replications_this_day,
                    label='Distinct Inflow',
                    color='red',
                    density=True,
                    bins=5
                    )

            ax.axvline(inflow_ground_truth, color='black', linestyle='dashed', label='Manually Observed')

            fig.subplots_adjust(bottom=.2)
            ax.set_title(
                'Estimate of distinct inflow of building {} \n'
                'for {} 1-4pm as of {}'.format(building,
                                               self.ground_truth_data_range[0][:10],
                                               str(self.end_of_backwards_window))

            )

            ax.set_xlabel('Distinct Inflow')
            ax.set_ylabel('Probability')
            ax.set_xlim(left=0)

            ax.legend()

            fig.tight_layout()

            fig.savefig(
                '{}/{}/{}_{}_{}_asof_{}_transition.png'.format(self.plot_location, 'histogram', building,
                                                               'inflow', self.ground_truth_data_range[0][:10],
                                                               str(self.end_of_backwards_window)
                                                               ))

            """
			OUTFLOW
			"""

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.hist(outflow_replications_this_day,
                    label='Distinct Outflow',
                    color='green',
                    density=True,
                    bins=5
                    )

            ax.axvline(outflow_ground_truth, color='black', linestyle='dashed', label='Manually Observed')

            fig.subplots_adjust(bottom=.2)
            ax.set_title(
                'Estimate of distinct outflow of building {} \n'
                'for {} 1-4pm as of {}'.format(building,
                                               self.ground_truth_data_range[0][:10],
                                               str(self.end_of_backwards_window)
                                               )
            )

            ax.set_xlabel('Distinct Outflow')
            ax.set_ylabel('Probability')
            ax.set_xlim(left=0)

            ax.legend()

            fig.tight_layout()

            fig.savefig(
                '{}/{}/blg_{}_{}_{}_asof_{}_transition.png'.format(self.plot_location, 'histogram', building,
                                                                   'outflow', self.ground_truth_data_range[0][:10],
                                                                   str(
                                                                       self.end_of_backwards_window)
                                                                   ))

            """
			OCCUPANCY
			"""

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.hist(occupancy_replications_this_day,
                    label='Distinct Occupancy',
                    color='blue',
                    density=True,
                    bins=5
                    )

            fig.subplots_adjust(bottom=.2)
            ax.set_title(
                'Estimate of distinct occupancy of building {} \n'
                'for {} 1-4pm as of {}'.format(building,
                                               self.ground_truth_data_range[0][:10],
                                               str(self.end_of_backwards_window))
            )

            ax.set_xlabel('Distinct Occupancy')
            ax.set_ylabel('Probability')
            ax.set_xlim(left=0)

            ax.legend()

            fig.tight_layout()

            fig.savefig(
                '{}/{}/blg_{}_{}_{}_asof_{}_transition.png'.format(self.plot_location, 'histogram', building,
                                                                   'occupancy', self.ground_truth_data_range[0][:10],
                                                                   str(
                                                                       self.end_of_backwards_window)
                                                                   ))

        return inflow_replications_this_day, outflow_replications_this_day  # not used currently

    def plot_observed_building_hourly_arrivals(self):

        start = time.time()
        self.logger.info(stringify('[plot_observed_building_hourly_arrivals] plotting observed inflow'))

        if self.limited_buiding_list == True:
            building_to_iterate_over = copy.deepcopy(self.debug_building_list)
        else:
            building_to_iterate_over = copy.deepcopy(self.card_reader_buildings_list)

        building_to_iterate_over.append('campus')

        if not os.path.exists(self.plot_location):
            os.makedirs(self.plot_location)

        time_freq = 'per_hour'

        if not os.path.exists('{}/{}'.format(self.plot_location, time_freq)):
            os.makedirs('{}/{}'.format(self.plot_location, time_freq))

        for building in tqdm(building_to_iterate_over):

            if building == 'campus':
                this_building_data = self.card_reader_df
                this_building_data = this_building_data[['swipes', 'timestamp']]
            else:
                this_building_data = self.card_reader_df[self.card_reader_df['location'] == building]
                this_building_data = this_building_data[['swipes', 'timestamp']]

            this_building_data_agg = this_building_data.groupby('timestamp').agg(['sum'])
            this_building_data_agg.reset_index(inplace=True)
            this_building_data_agg.columns = ['timestamp', 'swipes']
            this_building_data_agg['timestamp'] = pd.to_datetime(this_building_data_agg['timestamp'])

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.plot(this_building_data_agg['timestamp'], this_building_data_agg['swipes'],
                    label='Sum of swipes',
                    color='red')

            ax.axvline(self.end_of_backwards_window, color='black', linestyle='dashed', label='Current time')

            ax.tick_params(axis='x', labelrotation=90)
            fig.subplots_adjust(bottom=.2)

            ax.set_title('Distinct outflow, distinct inflow, occupancy \n'
                         'estimates of building {} \n '.format(
                             building
                         ))

            date_form = DateFormatter('%b-%d')
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

            ax.set_xlabel('Datetime')
            ax.set_ylabel('Count')

            handles, labels = ax.get_legend_handles_labels()

            ax.legend(handles, labels, fontsize='x-small')

            fig.tight_layout()

            fig.savefig(
                '{}/{}/{}_observed_inflow_{}.png'.format(self.plot_location,
                                                         'per_hour',
                                                         building,
                                                         self.experiment_name
                                                         ))

        self.logger.info(stringify('[plot_observed_building_hourly_arrivals]', time.time() - start, '\n'))

    def plot_simulated_timeseries(self):
        start = time.time()
        self.logger.info(stringify('[plot_simulated_timeseries] saving to {}'.format(self.plot_location)))

        if self.limited_buiding_list == True:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = self.card_reader_buildings_list

        self.campus_daily_occupancy_stats_df = pd.DataFrame()
        self.campus_daily_inflow_stats_df = pd.DataFrame()

        self.campus_hourly_occupancy_stats_df = pd.DataFrame()
        self.campus_hourly_inflow_stats_df = pd.DataFrame()
        self.campus_hourly_outflow_stats_df = pd.DataFrame()

        for building in tqdm(building_to_iterate_over):
            self.render_simulated_building_hourly_plot(building, 'inflow', 'outflow', 'occupancy', 'per_hour')
            self.render_simulated_building_daily_plot(building, 'peak_daily_inflow', 'peak_daily_occupancy', 'per_day')

        self.logger.info(
            stringify('[plot_simulated_timeseries] timeseries plotting finished', time.time() - start, '\n'))

    def report_percent_non_naive_transitions(self):

        list_of_adjancency_matrix = [self.all_samples_list[r]['adjacency_matrix']
                                     for r in range(len(self.all_samples_list))]

        mean_adj = np.mean(list_of_adjancency_matrix, axis=0)

        self.logger.info(stringify('[report_percent_non_naive_transitions] % of non-naive transitions for {} :'.format(
            self.experiment_name),
            100 * (np.sum(mean_adj) - np.trace(mean_adj)) / np.sum(mean_adj)))

    def plot_simulated_building_to_building_histogram(self):

        import networkx as nx
        import matplotlib.pyplot as plt
        import pickle

        if not os.path.exists(self.plot_location):
            os.makedirs(self.plot_location)

        if not os.path.exists('{}/{}'.format(self.plot_location, 'transitions')):
            os.makedirs('{}/{}'.format(self.plot_location, 'transitions'))

        self.logger.info(stringify('plotting assignment bar plots..'))

        if self.limited_buiding_list == True:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = list(self.all_buildings_assignments_pmf.keys())

        for blg in tqdm(building_to_iterate_over):
            blg_data_dict = self.all_buildings_assignments_pmf[blg]
            blg_data_df = pd.DataFrame.from_dict(blg_data_dict,
                                                 orient='index',
                                                 columns=self.all_buildings)
            blg_data_df['swipe_building'] = blg

            if blg_data_df.shape[0] == 0:
                self.logger.info(stringify(blg, 'no data'))
                continue

            blg_data_agg = blg_data_df.groupby(['swipe_building']).agg(['mean'])
            blg_data_agg.reset_index(inplace=True)
            columns = list(blg_data_agg.columns.get_level_values(0))

            blg_data_agg.columns = columns
            blg_data_agg['swipe_building'] = blg_data_agg.index

            blg_data_agg_long = pd.melt(blg_data_agg, id_vars=['swipe_building'])
            blg_data_agg_long.columns = ['swipe_building', 'assigned_building', 'mean_probability']
            blg_data_agg_long = blg_data_agg_long.sort_values(by='mean_probability', ascending=False)

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.bar(
                list(blg_data_agg_long['assigned_building'].values)[:20],
                list(blg_data_agg_long['mean_probability'].values)[:20]
            )
            ax.tick_params(axis='x', labelrotation=90)
            fig.subplots_adjust(bottom=.2)

            ax.set_title("blg {}'s swipe's mean OBSERVED (over all previous dates) \n"
                         'assignment to other buildings'.format(
                             blg))
            ax.set_xlabel('Assigned building')
            ax.set_ylabel('Mean Probability')

            fig.savefig('{}/transitions/{}_assignments_bar_{}.png'.format(self.plot_location,
                                                                          blg, self.experiment_name))

        if self.limited_buiding_list == True:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = list(self.all_buildings_transition_pmf.keys())

        self.logger.info(stringify('plotting observed data transition'))

        for blg in tqdm(building_to_iterate_over):
            if blg in self.all_buildings_transition_pmf:
                blg_data_dict = self.all_buildings_transition_pmf[blg]
                blg_data_df = pd.DataFrame.from_dict(blg_data_dict,
                                                     orient='index',
                                                     columns=self.all_buildings)
                blg_data_df['swipe_building'] = blg

                if blg_data_df.shape[0] == 0:
                    self.logger.info(stringify(blg, 'no data'))
                    continue

                blg_data_agg = blg_data_df.groupby(['swipe_building']).agg(['mean'])
                blg_data_agg.reset_index(inplace=True)
                columns = list(blg_data_agg.columns.get_level_values(0))
                blg_data_agg.columns = columns

                blg_data_agg_long = pd.melt(blg_data_agg, id_vars=['swipe_building'])
                blg_data_agg_long.columns = ['swipe_building', 'assigned_building', 'mean_probability']
                blg_data_agg_long = blg_data_agg_long.sort_values(by='mean_probability', ascending=False)

                fig = Figure(dpi=300)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)

                ax.bar(
                    list(blg_data_agg_long['assigned_building'].values)[:20],
                    list(blg_data_agg_long['mean_probability'].values)[:20]
                )
                ax.tick_params(axis='x', labelrotation=90)
                fig.subplots_adjust(bottom=.2)

                ax.set_title("blg {}'s mean OBSERVED transition probability \n"
                             '(over all hour of week) to other buildings'.format(blg))
                ax.set_xlabel('Destination building')
                ax.set_ylabel('Mean Probability')

                fig.savefig('{}/transitions/{}_observed_transitions_{}.png'.format(self.plot_location,
                                                                                   blg, self.experiment_name))
        self.logger.info(stringify('plotting simulated transition bar charts..'))

        list_of_adjancency_matrix = [self.all_samples_list[r]['adjacency_matrix']
                                     for r in range(len(self.all_samples_list))]
        mean_adj = np.mean(list_of_adjancency_matrix, axis=0)

        if self.limited_buiding_list == True:
            building_to_iterate_over = self.debug_building_list
        else:
            building_to_iterate_over = self.card_reader_buildings_list

        for building in tqdm(building_to_iterate_over):
            blg_index = self.all_buildings.index(building)

            transition_building_counts = mean_adj[blg_index]

            if sum(transition_building_counts) == 0:
                continue

            freq_dict = {}
            for i in range(len(self.all_buildings)):
                freq_dict[self.all_buildings[i]] = transition_building_counts[i]

            freq_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}

            buildings_to_plot = list(freq_dict.items())[:20]

            names = [i[0] for i in buildings_to_plot]
            counts = [i[1] for i in buildings_to_plot]

            fig = Figure(dpi=300)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            ax.bar(
                range(len(names)),
                counts
            )
            ax.set_xticklabels(names)
            ax.set_xticks(np.arange(len(names)))
            ax.tick_params(axis='x', labelrotation=90)
            fig.subplots_adjust(bottom=.2)

            ax.set_title("blg {}'s counts of SIMULATED transition \n"
                         'to other buildings'.format(building))
            ax.set_xlabel('Work building')
            ax.set_ylabel('Transitions')

            fig.savefig('{}/transitions/{}_simulated_transitions_{}.png'.format(self.plot_location,
                                                                                building, self.experiment_name))

    def render_simulated_building_daily_plot(self, building, metric1, metric2, time_freq):

        temp_inflow_data = pd.DataFrame.from_dict(self.output_json['simulation'][metric1][time_freq][building],
                                                  orient='index',
                                                  columns=['mean', 'std', '10 percentile', '50 percentile',
                                                           '90 percentile'])

        temp_inflow_data['timestamp'] = temp_inflow_data.index
        temp_inflow_data['timestamp'] = pd.to_datetime(temp_inflow_data['timestamp'])

        temp_campus_inflow_data = temp_inflow_data[['timestamp', 'mean', '50 percentile']]
        self.campus_daily_inflow_stats_df = pd.concat([self.campus_daily_inflow_stats_df,
                                                       temp_campus_inflow_data])

        temp_occupancy_data = pd.DataFrame.from_dict(self.output_json['simulation'][metric2][time_freq][building],
                                                     orient='index',
                                                     columns=['mean', 'std', '10 percentile', '50 percentile',
                                                              '90 percentile'])
        temp_occupancy_data['timestamp'] = temp_occupancy_data.index
        temp_occupancy_data['timestamp'] = pd.to_datetime(temp_occupancy_data['timestamp'])
        temp_campus_occupancy_data = temp_occupancy_data[['timestamp', 'mean', '50 percentile']]
        self.campus_daily_occupancy_stats_df = pd.concat([self.campus_daily_occupancy_stats_df,
                                                          temp_campus_occupancy_data])

        fig = Figure(dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(temp_inflow_data['timestamp'], temp_inflow_data['50 percentile'],
                label='Median {}'.format(metric1.replace('_', ' ').title()),
                color='red')
        ax.fill_between(temp_inflow_data['timestamp'],
                        temp_inflow_data['10 percentile'],
                        temp_inflow_data['90 percentile'],
                        color='red', alpha=0.5,
                        label='10-90% {}'.format(metric1.replace('_', ' ').title())
                        )

        ax.plot(temp_occupancy_data['timestamp'], temp_occupancy_data['50 percentile'],
                label='Median {}'.format(metric2.replace('_', ' ').title()),
                color='blue')
        ax.fill_between(temp_occupancy_data['timestamp'],
                        temp_occupancy_data['10 percentile'],
                        temp_occupancy_data['90 percentile'],
                        color='blue', alpha=0.5,
                        label='10-90% {}'.format(metric2.replace('_', ' ').title())
                        )

        ax.axvline(self.end_of_backwards_window, color='black', linestyle='dashed', label='Current time')

        ax.tick_params(axis='x', labelrotation=90)
        fig.subplots_adjust(bottom=.2)
        ax.set_title("Building {}'s peak daily inflow and occupancy".format(building))
        date_form = DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

        ax.set_xlabel('Date')
        ax.set_ylabel('Distinct Individuals')

        handles, labels = ax.get_legend_handles_labels()
        handles = [
            handles[2],
            handles[0],
            handles[3],
            handles[1],
            handles[4]
        ]
        labels = [
            labels[2],
            labels[0],
            labels[3],
            labels[1],
            labels[4]
        ]

        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')

        if not os.path.exists(self.plot_location):
            os.makedirs(self.plot_location)

        if not os.path.exists('{}/{}'.format(self.plot_location, time_freq)):
            os.makedirs('{}/{}'.format(self.plot_location, time_freq))

        fig.tight_layout()

        fig.savefig('{}/{}/{}_simulated_{}.png'.format(self.plot_location, time_freq, building,
                                                       self.experiment_name))

    def render_simulated_campus_hourly_plot(self, metric1, metric2, metric3, time_freq):

        temp_inflow_data = self.campus_hourly_inflow_stats_df
        temp_inflow_data = temp_inflow_data.groupby('timestamp').agg(['sum'])
        temp_inflow_data.reset_index(inplace=True)
        temp_inflow_data.columns = self.campus_hourly_inflow_stats_df.columns
        temp_inflow_data['timestamp'] = pd.to_datetime(temp_inflow_data['timestamp'])

        temp_outflow_data = self.campus_hourly_outflow_stats_df
        temp_outflow_data = temp_outflow_data.groupby('timestamp').agg(['sum'])
        temp_outflow_data.reset_index(inplace=True)
        temp_outflow_data.columns = self.campus_hourly_outflow_stats_df.columns
        temp_outflow_data['timestamp'] = pd.to_datetime(temp_outflow_data['timestamp'])

        temp_occupancy_data = self.campus_hourly_occupancy_stats_df
        temp_occupancy_data = temp_occupancy_data.groupby('timestamp').agg(['sum'])
        temp_occupancy_data.reset_index(inplace=True)
        temp_occupancy_data.columns = self.campus_hourly_occupancy_stats_df.columns
        temp_occupancy_data['timestamp'] = pd.to_datetime(temp_occupancy_data['timestamp'])

        fig = Figure(dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        date_form = DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

        ax.plot(temp_inflow_data['timestamp'], temp_inflow_data['50 percentile'],
                label='Median Distinct {}'.format(metric1.replace('_', ' ').title()),
                color='red', alpha=0.5)

        ax.plot(temp_outflow_data['timestamp'], temp_outflow_data['50 percentile'],
                label='Median Distinct {}'.format(metric2.replace('_', ' ').title()),
                color='green', alpha=0.5)

        ax.plot(temp_occupancy_data['timestamp'], temp_occupancy_data['50 percentile'],
                label='Median Distinct {}'.format(metric3.replace('_', ' ').title()),
                color='blue', alpha=0.5)

        ax.axvline(self.end_of_backwards_window, color='black', linestyle='dashed', label='Current time')

        ax.tick_params(axis='x', labelrotation=90)
        fig.subplots_adjust(bottom=.2)

        ax.set_title('Distinct outflow, distinct inflow, occupancy \n'
                     'estimates of campus \n '
                     'as of {}'.format(
                         str(self.end_of_backwards_window))
                     )

        ax.set_xlabel('Datetime')
        ax.set_ylabel('Count')
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(handles, labels, fontsize='x-small')

        if not os.path.exists(self.plot_location):
            os.makedirs(self.plot_location)

        if not os.path.exists('{}/{}'.format(self.plot_location, time_freq)):
            os.makedirs('{}/{}'.format(self.plot_location, time_freq))

        fig.tight_layout()

        fig.savefig(
            '{}/{}/{}_asof_{}_transition_{}.png'.format(self.plot_location, time_freq, 'campus',
                                                        str(self.end_of_backwards_window),
                                                        self.experiment_name
                                                        ))

    def render_simulated_building_hourly_plot(self, building, metric1, metric2, metric3, time_freq):

        temp_inflow_data = pd.DataFrame.from_dict(self.output_json['simulation'][metric1][time_freq][building],
                                                  orient='index',
                                                  columns=['mean', 'std', '10 percentile', '50 percentile',
                                                           '90 percentile'])

        temp_inflow_data['timestamp'] = temp_inflow_data.index
        temp_inflow_data['timestamp'] = pd.to_datetime(temp_inflow_data['timestamp'])
        temp_campus_inflow_data = temp_inflow_data[['timestamp', 'mean', '50 percentile']]
        self.campus_hourly_inflow_stats_df = pd.concat([self.campus_hourly_inflow_stats_df,
                                                        temp_campus_inflow_data])

        temp_outflow_data = pd.DataFrame.from_dict(self.output_json['simulation'][metric2][time_freq][building],
                                                   orient='index',
                                                   columns=['mean', 'std', '10 percentile', '50 percentile',
                                                            '90 percentile'])

        temp_outflow_data['timestamp'] = temp_outflow_data.index
        temp_outflow_data['timestamp'] = pd.to_datetime(temp_outflow_data['timestamp'])
        temp_campus_outflow_data = temp_outflow_data[['timestamp', 'mean', '50 percentile']]
        self.campus_hourly_outflow_stats_df = pd.concat([self.campus_hourly_outflow_stats_df,
                                                         temp_campus_outflow_data])

        temp_occupancy_data = pd.DataFrame.from_dict(self.output_json['simulation'][metric3][time_freq][building],
                                                     orient='index',
                                                     columns=['mean', 'std', '10 percentile', '50 percentile',
                                                              '90 percentile'])
        temp_occupancy_data['timestamp'] = temp_occupancy_data.index
        temp_occupancy_data['timestamp'] = pd.to_datetime(temp_occupancy_data['timestamp'])
        temp_campus_occupancy_data = temp_occupancy_data[['timestamp', 'mean', '50 percentile']]
        self.campus_hourly_occupancy_stats_df = pd.concat([self.campus_hourly_occupancy_stats_df,
                                                           temp_campus_occupancy_data])

        fig = Figure(dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        max_occ = np.max(temp_occupancy_data[
            (temp_occupancy_data['timestamp'] > self.end_of_backwards_window -
             timedelta(hours=self.plot_n_hours_window_backward))
            &
            (temp_occupancy_data['timestamp'] < self.end_of_backwards_window +
             timedelta(hours=self.plot_n_hours_window_forward))
        ]['90 percentile'])

        max_out = np.max(temp_outflow_data[
            (temp_outflow_data['timestamp'] > self.end_of_backwards_window -
             timedelta(hours=self.plot_n_hours_window_backward))
            &
            (temp_outflow_data['timestamp'] < self.end_of_backwards_window +
             timedelta(hours=self.plot_n_hours_window_forward))
        ]['90 percentile'])

        max_inf = np.max(temp_inflow_data[
            (temp_inflow_data['timestamp'] > self.end_of_backwards_window -
             timedelta(hours=self.plot_n_hours_window_backward))
            &
            (temp_inflow_data['timestamp'] < self.end_of_backwards_window +
             timedelta(hours=self.plot_n_hours_window_forward))
        ]['90 percentile'])

        max_y_axis = np.max([max_occ, max_out, max_inf])

        ax.set_xlim(self.end_of_backwards_window - timedelta(hours=self.plot_n_hours_window_backward),
                    self.end_of_backwards_window + timedelta(hours=self.plot_n_hours_window_forward))

        ax.plot(temp_inflow_data['timestamp'], temp_inflow_data['50 percentile'],
                label='Median Distinct {}'.format(metric1.replace('_', ' ').title()),
                alpha=0.5,
                linewidth=0.5,
                color='red')
        ax.fill_between(temp_inflow_data['timestamp'],
                        temp_inflow_data['10 percentile'],
                        temp_inflow_data['90 percentile'],
                        color='red', alpha=0.2,
                        label='10-90% Distinct {}'.format(metric1.replace('_', ' ').title())
                        )

        ax.plot(temp_outflow_data['timestamp'], temp_outflow_data['50 percentile'],
                label='Median Distinct {}'.format(metric2.replace('_', ' ').title()),
                alpha=0.5,
                linewidth=0.5,
                color='green')
        ax.fill_between(temp_outflow_data['timestamp'],
                        temp_outflow_data['10 percentile'],
                        temp_outflow_data['90 percentile'],
                        color='green', alpha=0.2,
                        label='10-90% Distinct {}'.format(metric2.replace('_', ' ').title())
                        )

        ax.plot(temp_occupancy_data['timestamp'], temp_occupancy_data['50 percentile'],
                label='Median Distinct {}'.format(metric3.replace('_', ' ').title()),
                alpha=0.5,
                linewidth=0.5,
                color='blue')
        ax.fill_between(temp_occupancy_data['timestamp'],
                        temp_occupancy_data['10 percentile'],
                        temp_occupancy_data['90 percentile'],
                        color='blue', alpha=0.2,
                        label='10-90% Distinct {}'.format(metric3.replace('_', ' ').title())
                        )

        ax.axvline(self.end_of_backwards_window, color='black', linestyle='dashed', label='Current time')

        ax.tick_params(axis='x', labelrotation=90)
        fig.subplots_adjust(bottom=.2)

        if self.GROUND_TRUTHING:
            ax.set_title('Distinct outflow, distinct inflow, occupancy \n'
                         'estimates of building {} for {} 1-4pm \n'
                         'as of {}'.format(
                             building,
                             self.ground_truth_data_range[0][:10],
                             str(self.end_of_backwards_window))
                         )
        else:
            ax.set_title('Distinct outflow, distinct inflow, occupancy \n'
                         'estimates of building {} \n '
                         'as of {}'.format(
                             building,
                             str(self.end_of_backwards_window))
                         )

        ax.set_ylim(0, max_y_axis)

        date_form = DateFormatter('%b %d %Hh')
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))

        ax.set_xlabel('Datetime')
        ax.set_ylabel('Count')

        handles, labels = ax.get_legend_handles_labels()

        handles = [
            handles[3],
            handles[0],
            handles[4],
            handles[1],
            handles[5],
            handles[2],
            handles[6]
        ]
        labels = [
            labels[3],
            labels[0],
            labels[4],
            labels[1],
            labels[5],
            labels[2],
            labels[6]
        ]

        ax.legend(handles, labels, fontsize='5')

        ax.grid(b=True, which='major', color='#666666', linestyle='-')
        ax.minorticks_on()
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        if not os.path.exists(self.plot_location):
            os.makedirs(self.plot_location)

        if not os.path.exists('{}/{}'.format(self.plot_location, time_freq)):
            os.makedirs('{}/{}'.format(self.plot_location, time_freq))

        fig.tight_layout()

        if self.GROUND_TRUTHING:
            fig.savefig(
                '{}/{}/{}_{}_asof_{}_transition.png'.format(self.plot_location, time_freq, building,
                                                            self.ground_truth_data_range[0][:10],
                                                            str(self.end_of_backwards_window)))
        else:
            fig.savefig(
                '{}/{}/{}_asof_{}_simulated_{}.png'.format(self.plot_location, time_freq, building,
                                                           str(self.end_of_backwards_window),
                                                           self.experiment_name
                                                           ))

    def write_output_n_metadata_to_disk(self):
        self.output_filename = self.uuid_prefix + '_results.json'
        self.output_filepath = self.io_dir + '/' + self.output_filename
        with open(self.output_filepath, 'w') as outfile:
            json.dump(self.output_json, outfile, indent=4)
        self.logger.info(stringify('Storing file locally: ' + self.output_filepath))

        self.metadata_filename = self.uuid_prefix + '_metadata.json'
        self.metadata_filepath = self.io_dir + '/' + self.metadata_filename
        with open(self.metadata_filepath, 'w') as outfile:
            json.dump(self.config, outfile, indent=4)
        self.logger.info(stringify('Storing file locally: ' + self.metadata_filepath))

    def cleanup(self):
        shutil.rmtree(self.io_dir)
        shutil.rmtree(self.log_file_location)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run individual analysis')
    parser.add_argument('--config-file', help='config filename', required=True)
    args = parser.parse_args()

    config_filepath = args.config_file

    simulation = SituationalAwareness(config_filepath)
    simulation.run()
