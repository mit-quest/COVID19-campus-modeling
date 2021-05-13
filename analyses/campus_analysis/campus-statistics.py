import os
import json
import datetime
import multiprocessing
import random
import copy
import time
import warnings

import pandas as pd
import numpy as np

from datetime import date, timedelta
from pathlib import Path

from models.common.mit_buildings import MITBuildings
from models.common.to_precision import sig_fig_formatted
from analyses.common.analysis import Analysis, add_empty_buildings, sort_dict


def read_row(pair_person_dates):
    '''
    function outputs a pd series (row). this row is then used by a multiprocessing pool join to
    make a pandas dataframe. this is done as a multprocessed fashion to speed up convering the json into a pandas dataframe.

    input is a list [person, dates] where:
        person: list of size <num. n_samples of people samples> which contains all the trajectory samples
            (i.e. input_samples['trajectory']['samples'])
        dates: list of size <num. of days n_days_to_simulate> of identically copied lists of dates from the samples
            (i.e. input_samples['trajectory']['dates'])
    '''

    building_stays_local = pd.DataFrame()
    coming_into_campus_local = pd.DataFrame()
    leaving_campus_local = pd.DataFrame()

    person = pair_person_dates[0]
    dates = pair_person_dates[1]
    id = pair_person_dates[2]

    for i, day_sample in enumerate(person):
        if len(day_sample) == 0:
            continue  # sampled person did not go to work
        else:
            for stay in day_sample:
                if stay['stay_type'] == 'on_campus_inside':
                    building_stays_local = building_stays_local.append(
                        {
                            'person_id': id,
                            'building': stay['building'],
                            'date': dates[i],
                            'start_time': dates[i] + ' ' + stay['start_time'],
                            'end_time': dates[i] + ' ' + stay['end_time']
                        }, ignore_index=True)

                # getting arrival time into campus to calculate campus level inflow down the line
                elif stay['stay_type'] == 'commute' and 'arrival_time' in stay.keys():
                    coming_into_campus_local = coming_into_campus_local.append(
                        {
                            'person_id': id,
                            'commute_type': stay['commute_type'],
                            'date': dates[i],
                            'arrival_time': dates[i] + ' ' + stay['arrival_time']
                        }, ignore_index=True)

                # getting arrival time into campus to calculate campus level inflow down the line
                elif stay['stay_type'] == 'commute' and 'departure_time' in stay.keys():
                    leaving_campus_local = leaving_campus_local.append(
                        {
                            'person_id': id,
                            'commute_type': stay['commute_type'],
                            'date': dates[i],
                            'departure_time': dates[i] + ' ' + stay['departure_time']
                        }, ignore_index=True)

    return {'building_stays_local': building_stays_local,
            'coming_into_campus_local': coming_into_campus_local,
            'leaving_campus_local': leaving_campus_local
            }


class ScenarioCampusStatistics(Analysis):

    def run(self, input_samples: dict, input_analyses: dict, uuid_prefix: str) -> dict:

        start_global = time.time()

        np.random.seed(self.analysis_parameters['random_seed'])

        if 'unittest_mode' not in self.analysis_parameters:
            self.unittest_mode = False
        else:
            self.unittest_mode = self.analysis_parameters['unittest_mode']

        self.n_bootstraps = self.analysis_parameters['n_bootstraps']
        self.percentiles_list = self.analysis_parameters['percentiles']

        dates = input_samples['trajectory']['dates']
        all_person_samples = input_samples['trajectory']['samples']

        self.num_samples = len(all_person_samples)

        self.building_data = MITBuildings()

        self.total_population_size = input_samples['trajectory']['total_population_size']

        cpu_count = multiprocessing.cpu_count()
        print('cpu count on machine:', cpu_count)

        dates_copies = [dates for copies in range(len(all_person_samples))]
        person_id_list = list(range(self.num_samples))
        input_to_multiprocessing = list(zip(all_person_samples, dates_copies, person_id_list))

        # read_row(input_to_multiprocessing[0]) #DEBUG for multiproc
        start_reading = time.time()
        print('creating reading multiprocessing pool..', cpu_count)
        pool = multiprocessing.Pool(cpu_count)
        print('reading..')
        reading_stays_list = pool.map(read_row, input_to_multiprocessing)
        pool.close()
        pool.join()
        print('reading finished, closed pool, took', time.time() - start_reading, 'sec')

        building_stays = pd.concat([ps['building_stays_local'] for ps in reading_stays_list])
        campus_arrivals = pd.concat([ps['coming_into_campus_local'] for ps in reading_stays_list])
        campus_depatures = pd.concat([ps['leaving_campus_local'] for ps in reading_stays_list])

        # print(building_stays.head()) #DEBUG
        # print(building_stays.shape) #DEBUG

        del input_to_multiprocessing

        print('converting timestamps')
        building_stays['date'] = pd.to_datetime(building_stays['date'])
        building_stays['start_time'] = pd.to_datetime(building_stays['start_time'])
        building_stays['end_time'] = pd.to_datetime(building_stays['end_time'])
        self.building_stays = building_stays

        campus_arrivals['date'] = pd.to_datetime(campus_arrivals['date'])
        campus_arrivals['arrival_time'] = pd.to_datetime(campus_arrivals['arrival_time'])
        self.campus_arrivals = campus_arrivals

        campus_depatures['date'] = pd.to_datetime(campus_depatures['date'])
        campus_depatures['departure_time'] = pd.to_datetime(campus_depatures['departure_time'])
        self.campus_depatures = campus_depatures

        # important normalizing factor
        self.stays_per_person = self.building_stays.shape[0] / self.num_samples
        # arrivals and departures can (should) normally be > 1 because a person has several arrivals departures over
        # different days
        self.arrivals_per_person = self.campus_arrivals.shape[0] / self.num_samples
        self.departures_per_person = self.campus_depatures.shape[0] / self.num_samples
        self.num_unique_people_in_all_samples = len(pd.unique(building_stays['person_id']))

        min_time = min(campus_arrivals['arrival_time'])
        min_time = min_time.replace(second=0, microsecond=0, minute=0, hour=min_time.hour)

        # max(campus_depatures['end_time']) to make sure we got that last time interval, adding + 1 to make sure
        # rounding doesn't make us miss the last interval
        max_time = max(campus_depatures['departure_time'])
        if max_time.hour == 23:
            max_time = max_time.replace(second=59, microsecond=0, minute=59, hour=max_time.hour)
        else:
            max_time = max_time.replace(second=0, microsecond=0, minute=0, hour=max_time.hour + 1)

        time_index_series = pd.date_range(
            min_time,
            max_time,
            freq='60min')

        self.time_demarkations = time_index_series.to_series().tolist()

        ######################## BOOTSTRAP ############

        # self.bootstrap(7)  # DEBUG
        start_boostrap = time.time()
        all_boostrap_data = pd.DataFrame()
        pool = multiprocessing.Pool(cpu_count)
        print('created bootstrap multiprocessing pool..', cpu_count)
        all_boostrap_data_list = pool.map(self.bootstrap, list(range(self.n_bootstraps)))
        pool.close()
        pool.join()

        # daily
        self.all_boostrap_campus_daily_sample_occupancy = pd.concat(
            [bs['daily_sample_occupancy'] for bs in all_boostrap_data_list])

        self.all_boostrap_campus_daily_sample_inflow = pd.concat(
            [bs['daily_sample_inflow'] for bs in all_boostrap_data_list])

        self.all_boostrap_campus_daily_sample_outflow = pd.concat(
            [bs['daily_sample_outflow'] for bs in all_boostrap_data_list])

        # hourly
        self.all_boostrap_inflow_data_campus_level_hourly = pd.concat(
            [bs['hourly_sample_inflow'] for bs in all_boostrap_data_list])

        self.all_boostrap_outflow_data_campus_level_hourly = pd.concat(
            [bs['hourly_sample_outflow'] for bs in all_boostrap_data_list])

        self.all_boostrap_occupancy_data_campus_level_hourly = pd.concat(
            [bs['hourly_sample_occupancy'] for bs in all_boostrap_data_list])

        print('bootstrapped finished, closed pool, took:', time.time() - start_boostrap, 'sec')

        # print('>>', self.all_boostrap_agg.head()) #DEBUG

        ######################## CREATING METRICS ############
        self.statistics_dict = {}

        self.day_list = pd.unique(self.all_boostrap_campus_daily_sample_occupancy['date'])
        self.day_list.sort()

        output_dict = {
            'simulation': {}
        }

        inside_dict = {
            'per_hour': {
                'all_buildings': {}
            },
            'per_day': {
                'all_buildings': {}
            }
        }

        for stat in ['inflow', 'outflow', 'distinct_individual_visits']:
            output_dict['simulation'][stat] = copy.deepcopy(inside_dict)

        """
        daily calculation
        """
        output_dict['simulation']['distinct_individual_visits']['per_day']['all_buildings'] = self.campus_stats_daily(
            self.all_boostrap_campus_daily_sample_occupancy, 'distinct_individual_visits')
        output_dict['simulation']['inflow']['per_day']['all_buildings'] = self.campus_stats_daily(
            self.all_boostrap_campus_daily_sample_outflow, 'outflow')
        output_dict['simulation']['outflow']['per_day']['all_buildings'] = self.campus_stats_daily(
            self.all_boostrap_campus_daily_sample_inflow, 'inflow')

        # """
        # hourly calculation
        # """
        output_dict['simulation']['distinct_individual_visits']['per_hour']['all_buildings'] = self.campus_stats_hourly(
            self.all_boostrap_occupancy_data_campus_level_hourly, 'distinct_individual_visits')
        output_dict['simulation']['inflow']['per_hour']['all_buildings'] = self.campus_stats_hourly(
            self.all_boostrap_inflow_data_campus_level_hourly, 'inflow')
        output_dict['simulation']['outflow']['per_hour']['all_buildings'] = self.campus_stats_hourly(
            self.all_boostrap_outflow_data_campus_level_hourly, 'outflow')

        print('>>>>> time taken to run all samples:', time.time()-start_global, 'sec')

        return sort_dict(output_dict)

    def campus_stats_hourly(self, dataframe: pd.DataFrame, metric: str):

        dates_dict = {}
        for i, interval in enumerate(self.time_demarkations):
            campus_this_hour = dataframe.loc[(dataframe['interval'] == interval)
                                             ]

            # calculating percentile for this building for this hour
            distribution = campus_this_hour[metric]

            if distribution.shape[0] > 0:
                percentiles_vals = np.percentile(distribution,
                                                 self.percentiles_list, interpolation='nearest')
            else:
                percentiles_vals = [None for i in range(len(self.percentiles_list))]

            # saving the simulated distribution
            simulated_output = {}

            for i, val in enumerate(self.percentiles_list):
                if percentiles_vals[i] is not None:
                    simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                else:
                    simulated_output[str(val) + ' percentile'] = 0

            dates_dict[str(interval)] = simulated_output
        print('hourly pmf for campus {} done'.format(metric))

        return sort_dict(dates_dict)

    def campus_stats_daily(self, dataframe: pd.DataFrame, metric: str):

        dates_dict = {}
        for day in self.day_list:
            campus_this_day = dataframe.loc[(dataframe['date'] == day)
                                            ]

            # calculating percentile for this building for this day
            distribution = campus_this_day[metric]

            if distribution.shape[0] > 0:
                percentiles_vals = np.percentile(distribution, self.percentiles_list, interpolation='nearest')
            else:
                percentiles_vals = [None for i in range(len(self.percentiles_list))]

            # saving the simulated distribution
            simulated_output = {}
            for i, val in enumerate(self.percentiles_list):
                if percentiles_vals[i] is not None:
                    simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                else:
                    simulated_output[str(val) + ' percentile'] = 0

            dates_dict[pd.to_datetime(str(day)).strftime('%Y-%m-%d 00:00:00')] = simulated_output
        print('daily pmf for {} done'.format(metric))

        return sort_dict(dates_dict)

    def bootstrap(self, seed: int):
        """
        we multiple the average num. of stays (or arrivals or depatures) per sample (people) x population size to
        get the number of bootstrap stays (or arrivals or depatures).
        """
        if self.num_unique_people_in_all_samples < self.total_population_size:
            warning_string = 'population being sampled (' + str(self.total_population_size) + \
                             ') is larger than sample size (' + str(self.num_unique_people_in_all_samples) + ')'
            warnings.warn(warning_string)

        campus_arrivals_boostrap = self.campus_arrivals.sample(
            n=int(self.arrivals_per_person * self.total_population_size),
            replace=True,
            random_state=seed
        )

        campus_departures_boostrap = self.campus_depatures.sample(
            n=int(self.departures_per_person * self.total_population_size),
            replace=True,
            random_state=seed
        )

        building_stays_boostrap = self.building_stays.sample(
            n=int(self.stays_per_person * self.total_population_size),
            replace=True,
            random_state=seed
        )

        """
        hourly unique occupancy calculation
        """

        # this is the list of times over which the trajectories will be binned, and then a daily mean and max
        # over these demarkations/times will be calculated
        time_demarkations = self.time_demarkations

        # hourly output dataframes
        hourly_sample_inflow = pd.DataFrame()
        hourly_sample_outflow = pd.DataFrame()
        hourly_sample_occupancy = pd.DataFrame()

        for i, interval in enumerate(time_demarkations):
            # this condition is to make sure we don't try to get the next demarkation if we're already at the end
            # unfortnuately means the last demarkation is thrown away
            if i + 1 != len(time_demarkations):
                """
                INFLOW
                """
                # condition below is basically checking: t_i <= start_time < t_(i+1) i.e. start_time was in this
                # interval t_i
                campus_level_inflow = campus_arrivals_boostrap.loc[  # start_time happens in this interval
                    (campus_arrivals_boostrap['arrival_time'] >= time_demarkations[i]) &
                    (campus_arrivals_boostrap['arrival_time'] < time_demarkations[i + 1])]

                # if trajectories had multiple arrivals
                if campus_level_inflow.shape[0] > 0:
                    hourly_sample_inflow = hourly_sample_inflow.append({
                        'inflow': campus_level_inflow.shape[0],
                        'day': interval.date(),
                        'interval': interval,
                    }, ignore_index=True)

                """
                OUTFLOW
                """
                # condition below is basically checking: t_i <= end_time < t_(i+1) i.e. end_time was in this
                # interval t_i
                campus_level_outflow = campus_departures_boostrap.loc[  # end_time happens in this interval
                    (campus_departures_boostrap['departure_time'] >= time_demarkations[i]) &
                    (campus_departures_boostrap['departure_time'] < time_demarkations[i + 1])]

                if campus_level_outflow.shape[0] > 0:
                    hourly_sample_outflow = hourly_sample_outflow.append({
                        'outflow': campus_level_outflow.shape[0],
                        'day': interval.date(),
                        'interval': interval,
                    }, ignore_index=True)

                """
                OCCUPANCY COUNTS
                """
                campus_level_selected_rows_start = building_stays_boostrap.loc[  # start_time happens in this interval
                    (building_stays_boostrap['start_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['start_time'] < time_demarkations[i + 1])]

                campus_level_selected_rows_end = building_stays_boostrap.loc[  # end_time happens in this interval
                    (building_stays_boostrap['end_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] < time_demarkations[i + 1])]

                campus_level_selected_rows_open = building_stays_boostrap.loc[  # start was before and end was after
                    # this interval
                    (building_stays_boostrap['start_time'] < time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] > time_demarkations[i + 1])]

                occupancy_interval = pd.concat([
                    campus_level_selected_rows_start,
                    campus_level_selected_rows_end,
                    campus_level_selected_rows_open
                ])

                if occupancy_interval.shape[0] > 0:
                    hourly_sample_occupancy = hourly_sample_occupancy.append({
                        'distinct_individual_visits': occupancy_interval.shape[0],
                        'day': interval.date(),
                        'interval': interval,
                    }, ignore_index=True)

        #     break #DEBUG
        # break #DEBUG

        """
        daily stats calculation
        """
        campus_daily_sample_inflow = campus_arrivals_boostrap['date']
        campus_daily_sample_inflow = campus_daily_sample_inflow.value_counts()
        campus_daily_sample_inflow = pd.DataFrame(campus_daily_sample_inflow)
        campus_daily_sample_inflow = campus_daily_sample_inflow.reset_index()
        campus_daily_sample_inflow.columns = ['date', 'inflow']

        campus_daily_sample_outflow = campus_departures_boostrap['date']
        campus_daily_sample_outflow = campus_daily_sample_outflow.value_counts()
        campus_daily_sample_outflow = pd.DataFrame(campus_daily_sample_outflow)
        campus_daily_sample_outflow = campus_daily_sample_outflow.reset_index()
        campus_daily_sample_outflow.columns = ['date', 'outflow']

        # using building_stays_boostrap here so that we get the people who are
        # currently on campus at this time. Note that there is no double counting
        # with campus_arrivals_boostrap and campus_departures_boostrap because these
        # do not contain any building level stays
        campus_daily_sample_occupancy = building_stays_boostrap['date']
        campus_daily_sample_occupancy = campus_daily_sample_occupancy.value_counts()
        campus_daily_sample_occupancy = pd.DataFrame(campus_daily_sample_occupancy)
        campus_daily_sample_occupancy = campus_daily_sample_occupancy.reset_index()
        campus_daily_sample_occupancy.columns = ['date', 'distinct_individual_visits']

        print('finished bootstrap:', seed, 'out of', self.n_bootstraps)

        return {
            # daily data
            'daily_sample_occupancy': campus_daily_sample_occupancy,
            'daily_sample_inflow': campus_daily_sample_inflow,
            'daily_sample_outflow': campus_daily_sample_outflow,

            # hourly data
            'hourly_sample_occupancy': hourly_sample_occupancy,
            'hourly_sample_inflow': hourly_sample_inflow,
            'hourly_sample_outflow': hourly_sample_outflow
        }
