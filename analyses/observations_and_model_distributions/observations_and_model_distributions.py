import os
import json
import datetime
import multiprocessing
import random
import copy
import time
import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from models.common.mit_buildings import MITBuildings
from models.common.to_precision import sig_fig_formatted
from analyses.common.analysis import Analysis, add_empty_buildings, sort_dict

pd.set_option('display.max_columns', None)


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

    return {'building_stays_local': building_stays_local,
            'coming_into_campus_local': coming_into_campus_local
            }


class ScenarioObservationsModelDistributions(Analysis):

    def run(self, input_samples: dict, input_analyses: dict, uuid_prefix: str) -> dict:

        start_global = time.time()

        np.random.seed(self.analysis_parameters['random_seed'])

        if 'unittest_mode' not in self.analysis_parameters:
            self.unittest_mode = False
        else:
            self.unittest_mode = self.analysis_parameters['unittest_mode']

        self.n_bootstraps = self.analysis_parameters['n_bootstraps']
        self.percentiles_list = self.analysis_parameters['percentiles']

        # loading all occupancy datasets_to_include:
        self.datasets_to_include = {}
        occupancy_dataset = {}
        for dataset in self.analysis_parameters['datasets_to_include']['occupancy']:
            occupancy = {}
            with open(dataset['path'], 'r') as fp:
                loaded_dataset = json.load(fp)
            occupancy['building_level'] = sort_dict(loaded_dataset[dataset['building_key_name']])
            occupancy['campus_level'] = sort_dict(loaded_dataset[dataset['campus_key_name']]['all'])
            occupancy_dataset[dataset['name']] = occupancy
            self.datasets_to_include['occupancy'] = occupancy_dataset

        # loading all inflow datasets_to_include:
        inflow_dataset = {}
        for dataset in self.analysis_parameters['datasets_to_include']['inflow']:
            inflow = {}
            with open(dataset['path'], 'r') as fp:
                loaded_dataset = json.load(fp)
            inflow['building_level'] = loaded_dataset[dataset['building_key_name']]
            inflow['campus_level'] = loaded_dataset[dataset['campus_key_name']]
            inflow_dataset[dataset['name']] = inflow
            self.datasets_to_include['inflow'] = inflow_dataset

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

        # print(building_stays.head()) #DEBUG
        # print(building_stays.shape) #DEBUG

        del input_to_multiprocessing
        del dates_copies
        del person_id_list

        print('converting timestamps')
        building_stays['date'] = pd.to_datetime(building_stays['date'])
        building_stays['end_time'] = pd.to_datetime(building_stays['end_time'])
        building_stays['start_time'] = pd.to_datetime(building_stays['start_time'])
        self.building_stays = building_stays

        campus_arrivals['date'] = pd.to_datetime(campus_arrivals['date'])
        campus_arrivals['arrival_time'] = pd.to_datetime(campus_arrivals['arrival_time'])
        self.campus_arrivals = campus_arrivals

        # important normalizing factor
        self.stays_per_person = self.building_stays.shape[0] / self.num_samples
        # arrivals and departures can (should) normally be > 1 because a person has several arrivals departures over
        # different days
        self.arrivals_per_person = self.campus_arrivals.shape[0] / self.num_samples
        self.num_unique_people_in_all_samples = len(pd.unique(building_stays['person_id']))

        min_time = min(building_stays['start_time'])
        min_time = min_time.replace(second=0, microsecond=0, minute=0, hour=min_time.hour)

        # max(building_stays['end_time']), # to make sure we got that last time interval, adding + 1 to make sure
        # rounding doesn't make us miss the last interval
        max_time = max(building_stays['end_time'])
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

        # daily data
        self.all_boostrap_daily_building_occupancy = pd.concat(
            [bs['daily_building_occupancy'] for bs in all_boostrap_data_list])

        self.all_boostrap_daily_building_inflow = pd.concat(
            [bs['daily_building_inflow'] for bs in all_boostrap_data_list])

        self.all_boostrap_daily_campus_occupancy = pd.concat(
            [bs['daily_campus_occupancy'] for bs in all_boostrap_data_list])

        self.all_boostrap_daily_campus_inflow = pd.concat(
            [bs['daily_campus_inflow'] for bs in all_boostrap_data_list])

        # hourly data
        self.all_boostrap_hourly_inflow_data_building_level = pd.concat(
            [bs['hourly_inflow_data_building_level'] for bs in all_boostrap_data_list])

        self.all_boostrap_hourly_inflow_data_campus_level = pd.concat(
            [bs['hourly_inflow_data_campus_level'] for bs in all_boostrap_data_list])

        self.all_boostrap_hourly_occupancy_data_campus_level = pd.concat(
            [bs['hourly_occupancy_data_campus_level'] for bs in all_boostrap_data_list])

        self.all_boostrap_hourly_occupancy_data_building_level = pd.concat(
            [bs['hourly_occupancy_data_building_level'] for bs in all_boostrap_data_list])

        print('bootstrapped finished, closed pool, took:', time.time() - start_boostrap, 'sec')

        ######################## CREATING METRICS ############
        self.statistics_dict = {}

        self.building_list = pd.unique(self.all_boostrap_daily_building_occupancy['building'])

        self.missing_buildings = list(set(self.building_data.known_building_ids()) - set(self.building_list))

        self.day_list = pd.unique(self.all_boostrap_daily_building_occupancy['date'])
        self.day_list.sort()

        self.building_dict = {}

        """
        calculations
        No outflow because data we have does not currently support outflow calculation e.g. building swipes
        from card reader only records people coming into a building, not leaving as people do not tap on their
        way out
        """

        # self.building_stats_daily(self.building_list[0])  # DEBUG
        # pool outputs a list of dictionaries, one item per building
        # TODO check why multiprocessing is not working fully. only 8-9 cores are being used instead of 96 (on a bi
        #  machine. could it be a memory read write issue? coz ram is not even close to maximized? )
        print('creating building_stats_daily analysis multiprocessing pool..', cpu_count)
        pool = multiprocessing.Pool(cpu_count)
        building_dict_list_daily = pool.map(self.building_stats_daily, self.building_list)
        pool.close()
        pool.join()
        print('building daily analysis finished, closed pool')

        # self.building_stats_hourly('13')  # DEBUG
        print('creating building_stats_hourly_inflow analysis multiprocessing pool..', cpu_count)
        pool = multiprocessing.Pool(cpu_count)
        building_dict_list_hourly = pool.map(self.building_stats_hourly, self.building_list)
        pool.close()
        pool.join()
        print('building hourly analysis finished, closed pool')

        # campus level
        campus_daily_data = self.campus_stats_daily()
        campus_hourly_data = self.campus_stats_hourly()

        for metric in ['occupancy', 'inflow']:
            """
            daily buildings
            """
            daily_buildings_data = {}
            for item in building_dict_list_daily:
                if len(item.keys()) > 0:
                    daily_buildings_data[item['building']] = item['building_stats_daily_output_dict'][metric]

            daily_buildings_data = add_empty_buildings(self.missing_buildings, daily_buildings_data)
            self.statistics_dict['building_daily_{}'.format(metric)] = daily_buildings_data

            """
            hourly buildings
            """
            hourly_buildings_data = {}
            for item in building_dict_list_hourly:
                if len(item.keys()) > 0:
                    hourly_buildings_data[item['building']] = item['building_stats_hourly_output_dict'][metric]

            hourly_buildings_data = add_empty_buildings(self.missing_buildings, hourly_buildings_data)
            self.statistics_dict['building_hourly_{}'.format(metric)] = hourly_buildings_data

            """
            daily campus
            """
            self.statistics_dict['campus_daily_{}'.format(metric)] = campus_daily_data[metric]

            """
            hourly campus
            """
            self.statistics_dict['campus_hourly_{}'.format(metric)] = campus_hourly_data[metric]

        print('>>>>> time taken to run all samples:', time.time()-start_global, 'sec')
        return sort_dict(self.statistics_dict)

    def campus_stats_hourly(self):

        # unique occupancy daily
        campus_stats_hourly_output_dict = {}

        for metric in ['occupancy', 'inflow']:
            metric_hourly_dict = {}

            if metric == 'occupancy':
                bootstrap_data_to_sample = self.all_boostrap_hourly_occupancy_data_campus_level
            elif metric == 'inflow':
                bootstrap_data_to_sample = self.all_boostrap_hourly_inflow_data_campus_level

            for i, interval in enumerate(self.time_demarkations):
                campus_this_hour = bootstrap_data_to_sample.loc[
                    (bootstrap_data_to_sample['interval'] == interval)
                ]

                # calculating percentile for this building for this day
                distribution = campus_this_hour[metric].value_counts(normalize=True)

                if distribution.shape[0] > 0:
                    distribution_df = pd.DataFrame(distribution)
                    distribution_df['value'] = distribution_df.index
                    distribution_df.columns = ['probability', 'value']

                    distribution_output = {}
                    for i in range(distribution_df.shape[0]):
                        row = distribution_df.iloc[i]
                        distribution_output[int(row['value'])] = row['probability']

                    distribution_mode = max(distribution_output, key=distribution_output.get)
                    percentiles_vals = np.percentile(
                        campus_this_hour[metric], self.percentiles_list, interpolation='nearest')

                else:
                    distribution_output = None
                    distribution_mode = 0

                    percentiles_vals = [None for i in range(len(self.percentiles_list))]

                # saving the simulated distribution
                all_data = {}
                simulated_output = {}

                for i, val in enumerate(self.percentiles_list):
                    if percentiles_vals[i] is not None:
                        simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                    else:
                        simulated_output[str(val) + ' percentile'] = 0

                simulated_output['pmf'] = distribution_output
                simulated_output['mode'] = distribution_mode

                # appending the data from different datasets
                datasets_to_include_temp = {}
                if metric in self.datasets_to_include.keys():
                    for dataset in self.datasets_to_include[metric].keys():
                        try:
                            value = self.datasets_to_include[metric][dataset]['campus_level'][str(interval)]
                        except KeyError:
                            # either building or date does not have a value
                            print('[hourly campus {}]'.format(metric), str(interval), 'not found in ', dataset)
                            value = None
                        datasets_to_include_temp[dataset] = value

                all_data['simulated'] = simulated_output
                all_data['datasets'] = datasets_to_include_temp

                metric_hourly_dict[str(interval)] = all_data

            print('hourly {} pmf for campus done'.format(metric))
            campus_stats_hourly_output_dict[metric] = sort_dict(metric_hourly_dict)

        return campus_stats_hourly_output_dict

    def building_stats_hourly(self, building):

        if self.building_data.is_valid_building_id(building) == False:
            print('invalid building:', building)
            return {}
        else:

            building_stats_hourly_output_dict = {}

            for metric in ['occupancy', 'inflow']:
                metric_hourly_dict = {}

                if metric == 'occupancy':
                    bootstrap_data_to_sample = self.all_boostrap_hourly_occupancy_data_building_level
                elif metric == 'inflow':
                    bootstrap_data_to_sample = self.all_boostrap_hourly_inflow_data_building_level

                for i, interval in enumerate(self.time_demarkations):
                    data_this_building_this_interval = bootstrap_data_to_sample.loc[
                        (bootstrap_data_to_sample['building'] == building)
                        &
                        (bootstrap_data_to_sample['interval'] == interval)
                    ]

                    # calculating percentile for this building for this day
                    distribution = data_this_building_this_interval[metric].value_counts(normalize=True)

                    if distribution.shape[0] > 0:
                        distribution_df = pd.DataFrame(distribution)
                        distribution_df['value'] = distribution_df.index
                        distribution_df.columns = ['probability', 'value']

                        distribution_output = {}
                        for i in range(distribution_df.shape[0]):
                            row = distribution_df.iloc[i]
                            distribution_output[int(row['value'])] = row['probability']

                        distribution_mode = max(distribution_output, key=distribution_output.get)
                        percentiles_vals = np.percentile(
                            data_this_building_this_interval[metric], self.percentiles_list, interpolation='nearest')

                    else:
                        distribution_output = None
                        distribution_mode = 0

                        percentiles_vals = [None for i in range(len(self.percentiles_list))]

                    # saving the simulated distribution
                    all_data = {}
                    simulated_output = {}

                    for i, val in enumerate(self.percentiles_list):
                        if percentiles_vals[i] is not None:
                            simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                        else:
                            simulated_output[str(val) + ' percentile'] = 0

                    simulated_output['pmf'] = distribution_output
                    simulated_output['mode'] = distribution_mode

                    # appending the data from different datasets
                    datasets_to_include_temp = {}
                    if metric in self.datasets_to_include.keys():
                        for dataset in self.datasets_to_include[metric].keys():
                            try:
                                value = self.datasets_to_include[metric][dataset]['building_level'][building][str(
                                    interval)]
                            except KeyError:
                                # either building or date does not have a value
                                print('[hourly {}]'.format(metric), building, str(interval), 'not found', dataset)
                                value = None
                            datasets_to_include_temp[dataset] = value

                    all_data['simulated'] = simulated_output
                    all_data['datasets'] = datasets_to_include_temp

                    metric_hourly_dict[str(interval)] = all_data

                print('hourly {} pmf for {} done'.format(metric, building))
                building_stats_hourly_output_dict[metric] = sort_dict(metric_hourly_dict)

            return {'building': building,
                    'building_stats_hourly_output_dict': building_stats_hourly_output_dict
                    }

    def building_stats_daily(self, building: str):

        building_stats_daily_output_dict = {}
        if self.building_data.is_valid_building_id(building) == False:
            print('invalid building:', building)
            return {}
        else:

            for metric in ['occupancy', 'inflow']:

                metric_daily_dict = {}
                all_hf_day_dict = {}

                if metric == 'occupancy':
                    bootstrap_data_to_sample = self.all_boostrap_daily_building_occupancy
                elif metric == 'inflow':
                    bootstrap_data_to_sample = self.all_boostrap_daily_building_inflow

                for day in self.day_list:
                    data_this_building_this_day = bootstrap_data_to_sample.loc[
                        (bootstrap_data_to_sample['building'] == building)
                        &
                        (bootstrap_data_to_sample['date'] == day)
                    ]

                    """
                    OCCUPANCY
                    """
                    # calculating percentile for this building for this day
                    distribution = data_this_building_this_day[metric].value_counts(normalize=True)

                    if distribution.shape[0] > 0:
                        distribution_df = pd.DataFrame(distribution)
                        distribution_df['value'] = distribution_df.index
                        distribution_df.columns = ['probability', 'value']

                        distribution_output = {}
                        for i in range(distribution_df.shape[0]):
                            row = distribution_df.iloc[i]
                            distribution_output[int(row['value'])] = row['probability']

                        distribution_mode = max(distribution_output, key=distribution_output.get)
                        percentiles_vals = np.percentile(
                            data_this_building_this_day[metric], self.percentiles_list, interpolation='nearest')

                    else:
                        distribution_output = None
                        distribution_mode = 0

                        percentiles_vals = [None for i in range(len(self.percentiles_list))]

                    # saving the simulated distribution
                    all_data = {}
                    simulated_output = {}

                    for i, val in enumerate(self.percentiles_list):
                        if percentiles_vals[i] is not None:
                            simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                        else:
                            simulated_output[str(val) + ' percentile'] = 0

                    simulated_output['pmf'] = distribution_output
                    simulated_output['mode'] = distribution_mode

                    # appending the data from different datasets
                    datasets_to_include_temp = {}
                    if metric in self.datasets_to_include.keys():
                        for dataset in self.datasets_to_include[metric].keys():
                            try:
                                value = self.datasets_to_include[metric][dataset]['building_level'][building][np.datetime_as_string(
                                    day, unit='D')]
                            except KeyError:
                                # either building or date does not have a value
                                print('[daily {}]'.format(metric), building, str(day), 'not found in', dataset)
                                value = None
                            datasets_to_include_temp[dataset] = value

                    all_data['simulated'] = simulated_output
                    all_data['datasets'] = datasets_to_include_temp

                    metric_daily_dict[np.datetime_as_string(day, unit='D')] = all_data

                building_stats_daily_output_dict[metric] = sort_dict(metric_daily_dict)
                print('daily {} pmf for {} done'.format(metric, building))

            return {
                'building': building,
                'building_stats_daily_output_dict': building_stats_daily_output_dict
            }

    def campus_stats_daily(self):

        campus_stats_daily_output_dict = {}

        for metric in ['occupancy', 'inflow']:
            metric_daily_dict = {}

            if metric == 'occupancy':
                bootstrap_data_to_sample = self.all_boostrap_daily_campus_occupancy
            elif metric == 'inflow':
                bootstrap_data_to_sample = self.all_boostrap_daily_campus_inflow

            for day in self.day_list:
                campus_this_day = bootstrap_data_to_sample.loc[
                    (bootstrap_data_to_sample['date'] == day)
                ]

                # calculating percentile for this building for this day
                distribution = campus_this_day[metric].value_counts(normalize=True)

                if distribution.shape[0] > 0:
                    distribution_df = pd.DataFrame(distribution)
                    distribution_df['value'] = distribution_df.index
                    distribution_df.columns = ['probability', 'value']

                    distribution_output = {}
                    for i in range(distribution_df.shape[0]):
                        row = distribution_df.iloc[i]
                        distribution_output[int(row['value'])] = row['probability']

                    distribution_mode = max(distribution_output, key=distribution_output.get)
                    percentiles_vals = np.percentile(
                        campus_this_day[metric], self.percentiles_list, interpolation='nearest')

                else:
                    distribution_output = None
                    distribution_mode = 0

                    percentiles_vals = [None for i in range(len(self.percentiles_list))]

                # saving the simulated distribution
                all_data = {}
                simulated_output = {}

                for i, val in enumerate(self.percentiles_list):
                    if percentiles_vals[i] is not None:
                        simulated_output[str(val) + ' percentile'] = int(percentiles_vals[i])
                    else:
                        simulated_output[str(val) + ' percentile'] = 0

                simulated_output['pmf'] = distribution_output
                simulated_output['mode'] = distribution_mode

                # appending the data from different datasets
                datasets_to_include_temp = {}
                if metric in self.datasets_to_include.keys():
                    for dataset in self.datasets_to_include[metric].keys():
                        try:
                            value = self.datasets_to_include[metric][dataset]['campus_level'][
                                np.datetime_as_string(day, unit='D')]
                        except KeyError:
                            # either building or date does not have a value
                            print('[daily campus {}]'.format(metric), np.datetime_as_string(day, unit='D'), 'not found in',
                                  dataset)
                            value = None
                        datasets_to_include_temp[dataset] = value

                all_data['simulated'] = simulated_output
                all_data['datasets'] = datasets_to_include_temp

                metric_daily_dict[np.datetime_as_string(day, unit='D')] = all_data

            campus_stats_daily_output_dict[metric] = sort_dict(metric_daily_dict)

            print('daily {} pmf for campus done'.format(metric))

        return campus_stats_daily_output_dict

    def bootstrap(self, seed: int):
        '''
        this function takes in a random seed, and then selects a number <self.building_stays.shape[0]> of
        stays from the trajectory samples to run a bootstrap. it then outputs the occupancy, inflow, outflow
        within each building for each time interval
        '''

        # building_stays_sample is the sampled (with replacement dataframe of trajectories)
        """
        n here is the avg. number of rows (i.e. stays) for the population.
        we first calculate the number of stays per sample (people) we have on avg, and
        multiply this by population size to get enough stays per person.
        this assumes each type of person has the same distribution over stays
        """
        if self.num_unique_people_in_all_samples < self.total_population_size:
            warning_string = 'population being sampled (' + str(self.total_population_size) + \
                             ') is larger than sample size (' + str(self.num_unique_people_in_all_samples) + ')'
            warnings.warn(warning_string)

        building_stays_boostrap = self.building_stays.sample(
            n=int(self.building_stays.shape[0] / self.num_samples * self.total_population_size),
            replace=True,
            random_state=seed
        )

        campus_arrival_bootstrap = self.campus_arrivals.sample(
            n=int(self.arrivals_per_person * self.total_population_size),
            replace=True,
            random_state=seed
        )
        """
        hourly unique occupancy calculation
        """
        # this is the list of times over which the trajectories will be binned, and then a daily mean and max
        # over these demarkations/times will be calculated
        time_demarkations = self.time_demarkations

        daily_inflow_data_building_level = pd.DataFrame()

        hourly_inflow_data_building_level = pd.DataFrame()
        hourly_inflow_data_campus_level = pd.DataFrame()

        hourly_occupancy_data_building_level = pd.DataFrame()
        hourly_occupancy_data_campus_level = pd.DataFrame()

        for i, interval in enumerate(time_demarkations):
            # this condition is to make sure we don't try to get the next demarkation if we're already at the end
            # unfortnuately means the last demarkation is thrown away TODO: include last time interval
            if i + 1 != len(time_demarkations):
                # building level getting stays
                # condition below is basically checking: t_i <= start_time < t_(i+1) i.e. start_time was in this
                # interval t_i
                building_level_selected_rows_start = building_stays_boostrap.loc[  # start_time happens in this interval
                    (building_stays_boostrap['start_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['start_time'] < time_demarkations[i + 1])]

                daily_inflow_data_building_level = pd.concat(
                    [daily_inflow_data_building_level, building_level_selected_rows_start])

                # condition below is basically checking: t_i <= end_time < t_(i+1) i.e. end_time was in this
                # interval t_i
                building_level_selected_rows_end = building_stays_boostrap.loc[  # end_time happens in this interval
                    (building_stays_boostrap['end_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] < time_demarkations[i + 1])]

                building_level_selected_rows_open = building_stays_boostrap.loc[  # start was before and end was after this interval
                    (building_stays_boostrap['start_time'] < time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] > time_demarkations[i + 1])]

                building_level_occupancy_interval = pd.concat([
                    building_level_selected_rows_start,
                    building_level_selected_rows_end,
                    building_level_selected_rows_open
                ])

                # campus level getting stays
                campus_level_selected_rows_start = building_stays_boostrap.loc[  # start_time happens in this interval
                    (building_stays_boostrap['start_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['start_time'] < time_demarkations[i + 1])]

                campus_level_selected_rows_end = building_stays_boostrap.loc[  # end_time happens in this interval
                    (building_stays_boostrap['end_time'] >= time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] < time_demarkations[i + 1])]

                # this one uses building stays because stays are from people being on campus (not just arriving or
                # leaving)
                campus_level_selected_rows_open = building_stays_boostrap.loc[  # start was before and end was after
                    # this interval
                    (building_stays_boostrap['start_time'] < time_demarkations[i]) &
                    (building_stays_boostrap['end_time'] > time_demarkations[i + 1])]

                campus_occupancy_interval = pd.concat([
                    campus_level_selected_rows_start,
                    campus_level_selected_rows_end,
                    campus_level_selected_rows_open
                ])

                """
                HOURLY OCCUPANCY COMPUTATION
                """
                # campus level
                if campus_occupancy_interval.shape[0] > 0:
                    hourly_occupancy_data_campus_level = hourly_occupancy_data_campus_level.append({
                        'occupancy': campus_occupancy_interval.shape[0],
                        'day': interval.date(),
                        'interval': interval,
                    }, ignore_index=True)

                # building level
                if building_level_occupancy_interval.shape[0] > 0:
                    occupancy_occupancy_column = building_level_occupancy_interval['building'].value_counts()
                    occupancy_occupancy_column_df = pd.DataFrame([occupancy_occupancy_column]).T
                    occupancy_occupancy_column_df.columns = ['occupancy']
                    occupancy_occupancy_column_df['building'] = occupancy_occupancy_column_df.index
                    occupancy_occupancy_column_df['day'] = interval.date()
                    occupancy_occupancy_column_df['interval'] = interval
                    hourly_occupancy_data_building_level = pd.concat(
                        [hourly_occupancy_data_building_level, occupancy_occupancy_column_df])

                """
                HOURLY INFLOW COMPUTATION
                """
                # campus level
                if campus_level_selected_rows_start.shape[0] > 0:
                    hourly_inflow_data_campus_level = hourly_inflow_data_campus_level.append({
                        'inflow': campus_level_selected_rows_start.shape[0],
                        'day': interval.date(),
                        'interval': interval,
                    }, ignore_index=True)

                # building level
                if building_level_selected_rows_start.shape[0] > 0:
                    # if building_level_selected_rows_start.shape[0] > 0:
                    building_level_selected_rows_start = building_level_selected_rows_start['building'].value_counts()
                    selected_rows_start_column_df = pd.DataFrame(
                        [building_level_selected_rows_start]).T
                    selected_rows_start_column_df.columns = ['inflow']
                    selected_rows_start_column_df['building'] = selected_rows_start_column_df.index
                    selected_rows_start_column_df['day'] = interval.date()
                    selected_rows_start_column_df['interval'] = interval

                    hourly_inflow_data_building_level = pd.concat([hourly_inflow_data_building_level,
                                                                   selected_rows_start_column_df])

        #     break #DEBUG
        # break #DEBUG

        """
        daily unique occupancy calculation
        """

        # building level unique occupancy per day (not hourly)
        building_daily_stays_sample = building_stays_boostrap[['building', 'date']]
        daily_building_occupancy = building_daily_stays_sample.value_counts()
        daily_building_occupancy = pd.DataFrame(daily_building_occupancy)
        daily_building_occupancy = daily_building_occupancy.reset_index()
        daily_building_occupancy.columns = ['building', 'date', 'occupancy']

        # campus level unique occupancy per day (not hourly)
        daily_campus_occupancy = building_stays_boostrap['date']
        daily_campus_occupancy = daily_campus_occupancy.value_counts()
        daily_campus_occupancy = pd.DataFrame(daily_campus_occupancy)
        daily_campus_occupancy = daily_campus_occupancy.reset_index()
        daily_campus_occupancy.columns = ['date', 'occupancy']

        """
        daily unique inflow calculation
        """
        building_daily_stays_sample = daily_inflow_data_building_level[['building', 'date']]
        daily_building_inflow = building_daily_stays_sample.value_counts()
        daily_building_inflow = pd.DataFrame(daily_building_inflow)
        daily_building_inflow = daily_building_inflow.reset_index()
        daily_building_inflow.columns = ['building', 'date', 'inflow']

        daily_campus_inflow = campus_arrival_bootstrap['date']
        daily_campus_inflow = daily_campus_inflow.value_counts()
        daily_campus_inflow = pd.DataFrame(daily_campus_inflow)
        daily_campus_inflow = daily_campus_inflow.reset_index()
        daily_campus_inflow.columns = ['date', 'inflow']

        print('finished bootstrap:', seed, 'out of', self.n_bootstraps)

        return {
            # daily data
            'daily_building_occupancy': daily_building_occupancy,
            'daily_building_inflow': daily_building_inflow,

            'daily_campus_occupancy': daily_campus_occupancy,
            'daily_campus_inflow': daily_campus_inflow,

            # hourly data
            'hourly_occupancy_data_building_level': hourly_occupancy_data_building_level,
            'hourly_inflow_data_building_level': hourly_inflow_data_building_level,

            'hourly_occupancy_data_campus_level': hourly_occupancy_data_campus_level,
            'hourly_inflow_data_campus_level': hourly_inflow_data_campus_level
        }
