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

    return building_stays_local


class ScenarioBuildingStatisticsAnalysis(Analysis):

    def run(self, input_samples: dict, input_analyses: dict, uuid_prefix: str) -> dict:

        start_global = time.time()

        np.random.seed(self.analysis_parameters['random_seed'])

        if 'unittest_mode' not in self.analysis_parameters:
            self.unittest_mode = False
        else:
            self.unittest_mode = self.analysis_parameters['unittest_mode']

        self.n_bootstraps = self.analysis_parameters['n_bootstraps']
        self.time_freq = self.analysis_parameters['time_freq']
        self.percentiles_list = self.analysis_parameters['percentiles']

        self.lunch_start = self.analysis_parameters['lunch_start']
        self.lunch_end = self.analysis_parameters['lunch_end']
        self.output_hourly = self.analysis_parameters['output_hourly']

        dates = input_samples['trajectory']['dates']
        all_person_samples = input_samples['trajectory']['samples']

        self.num_samples = len(all_person_samples)

        list_agg_type = ['mean', 'max']

        self.building_data = MITBuildings()

        self.total_population_size = input_samples['trajectory']['total_population_size']

        cpu_count = multiprocessing.cpu_count()
        print('cpu count on machine:', cpu_count)

        dates_copies = [dates for copies in range(len(all_person_samples))]
        person_id_list = list(range(self.num_samples))
        input_to_multiprocessing = list(zip(all_person_samples, dates_copies, person_id_list))

        print('creating reading multiprocessing pool..', cpu_count)
        pool = multiprocessing.Pool(cpu_count)
        print('reading..')
        building_stays = pd.concat(pool.map(read_row, input_to_multiprocessing))
        pool.close()
        pool.join()
        print('reading finished, closed pool')

        # print(building_stays.head()) #DEBUG
        # print(building_stays.shape) #DEBUG

        del input_to_multiprocessing
        del dates_copies
        del person_id_list

        print('converting timestamps')
        building_stays['date'] = pd.to_datetime(building_stays['date'])
        building_stays['end_time'] = pd.to_datetime(building_stays['end_time'])
        building_stays['start_time'] = pd.to_datetime(building_stays['start_time'])

        self.first_day = min(pd.to_datetime(building_stays['date']))

        self.building_stays = building_stays

        # important normalizing factor. self.building_stays only has in-building stays (i.e. stay_type ==
        # on_campus_inside)
        self.stays_per_person_per_simrange = self.building_stays.shape[0] / self.num_samples
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
            freq=self.time_freq)

        self.time_demarkations = time_index_series.to_series().tolist()

        ######################## BOOTSTRAP ############

        # self.bootstrap(7)  # DEBUG

        all_boostrap_data = pd.DataFrame()
        pool = multiprocessing.Pool(cpu_count)
        print('created bootstrap multiprocessing pool..', cpu_count)
        all_boostrap_data_list = pool.map(self.bootstrap, list(range(self.n_bootstraps)))
        pool.close()
        pool.join()

        self.all_boostrap_data = pd.concat([i['all_data'] for i in all_boostrap_data_list])
        self.all_boostrap_agg = pd.concat([i['all_aggs'] for i in all_boostrap_data_list])

        print('bootstrapped finished, closed pool')

        # print('>>', self.all_boostrap_agg.head()) #DEBUG

        ######################## CREATING METRICS ############

        self.statistics_dict = {
            'simulation': {}
        }

        inside_dict = {
            'per_hour': {
            },
            'per_day': {
            },
            'full_simulation_time_window': {
            }
        }

        for stat in ['inflow', 'outflow', 'distinct_individual_visits', 'density']:
            self.statistics_dict['simulation'][stat] = copy.deepcopy(inside_dict)

        self.building_list = pd.unique(self.all_boostrap_agg['building'])

        self.missing_buildings = list(set(self.building_data.known_building_ids()) - set(self.building_list))

        self.day_list = pd.unique(self.all_boostrap_agg['day'])
        self.day_list.sort()

        for agg_type in list_agg_type:

            self.compute_statistic(
                bootstrap_column='distinct_individual_visits',
                agg_type=agg_type,
                output_metric_name='distinct_individual_visits',
                density=False)

            if self.unittest_mode == True:
                break  # only test/run occupancy if in unittest_mode mode

            self.compute_statistic(
                bootstrap_column='inflow',
                agg_type=agg_type,
                output_metric_name='inflow',
                density=False)

            self.compute_statistic(
                bootstrap_column='outflow',
                agg_type=agg_type,
                output_metric_name='outflow',
                density=False)

            self.compute_statistic(
                bootstrap_column='distinct_individual_visits',
                agg_type=agg_type,
                output_metric_name='density',
                density=True)

        self.statistics_dict = sort_dict(self.statistics_dict)
        # print(self.statistics_dict) #DEBUG

        print('>>>>> time taken to run all samples:', time.time()-start_global, 'sec')

        return self.statistics_dict

    def compute_statistic(self,
                          bootstrap_column,
                          agg_type,
                          output_metric_name,
                          density):

        print('creating metric:', agg_type, bootstrap_column)
        self.building_dict = {}
        self.building_dict_agg_all_days = {}

        cpu_count = multiprocessing.cpu_count()

        self.bootstrap_column = bootstrap_column
        self.agg_type = agg_type
        self.output_metric_name = output_metric_name
        self.density = density

        # pool outputs a list of dictionaries, one item per building
        # TODO check why multiprocessing is not working fully. only 8-9 cores are being used instead of 96 (on a bi
        #  machine. could it be a memory read write issue? coz ram is not even close to maximized? )
        print('creating building analysis multiprocessing pool..', cpu_count)
        pool = multiprocessing.Pool(cpu_count)
        building_dict_list = pool.map(self.building_stats, self.building_list)
        pool.close()
        pool.join()
        print('analysis for {} {} finished, closed pool'.format(
            self.output_metric_name, self.agg_type))

        output_daily = {}
        output_all_days = {}
        output_all_day_hf = {}
        for item in building_dict_list:
            if len(item.keys()) > 0:
                output_daily[item['building']] = item['dates_dict']
                output_all_days[item['building']] = item['all_percentiles_dict']
                output_all_day_hf[item['building']] = item['all_hf_day_dict']

        output_daily = add_empty_buildings(self.missing_buildings, output_daily)
        output_all_days = add_empty_buildings(self.missing_buildings, output_all_days)
        output_all_day_hf = add_empty_buildings(self.missing_buildings, output_all_day_hf)

        self.statistics_dict['simulation'][self.output_metric_name]['per_day'] = output_daily
        self.statistics_dict['simulation'][self.output_metric_name]['full_simulation_time_window'] = output_all_days

        if self.output_hourly == True:
            self.statistics_dict['simulation'][self.output_metric_name]['per_hour'] = output_all_day_hf

    def bootstrap(self, seed: int):
        '''
        this function takes in a random seed, and then selects a number <self.building_stays.shape[0]> of
        stays from the trajectory samples to run a bootstrap. it then outputs the occupancy, inflow, outflow
        within each building for each time interval
        '''

        # building_stays_sample is the sampled (with replacement dataframe of trajectories)
        # TODO fix the sampling - it is assuming each type of person has the same distribution over stays
        # right now we're just using an average stays per person to set the number of samples below
        if self.num_unique_people_in_all_samples < self.total_population_size:
            warning_string = 'population being sampled (' + str(self.total_population_size) + \
                             ') is larger than sample size (' + str(self.num_unique_people_in_all_samples) + ')'
            warnings.warn(warning_string)

        building_stays_sample = self.building_stays.sample(
            n=int(self.stays_per_person_per_simrange * self.total_population_size),
            replace=True,
            random_state=seed
        )

        # this is the list of times over which the trajectories will be binned, and then a daily mean and max
        # over these demarkations/times will be calculated
        time_demarkations = self.time_demarkations

        occupancy_data = pd.DataFrame()
        inflow_data = pd.DataFrame()
        outflow_data = pd.DataFrame()

        for i, interval in enumerate(time_demarkations):
            # this condition is to make sure we don't try to get the next demarkation if we're already at the end
            # unfortnuately means the last demarkation is thrown away. the complexity of the code becomes too big to account for (so far)
            if i + 1 != len(time_demarkations):
                # condition below is basically checking: t_i <= start_time < t_(i+1) i.e. start_time was in this
                # interval t_i
                selected_rows_start = building_stays_sample.loc[  # start_time happens in this interval
                    (building_stays_sample['start_time'] >= time_demarkations[i]) &
                    (building_stays_sample['start_time'] < time_demarkations[i + 1])]
                if selected_rows_start.shape[0] > 0:
                    selected_rows_start_column = selected_rows_start['building'].value_counts()
                    selected_rows_start_column_df = pd.DataFrame(
                        [selected_rows_start_column]).T
                    selected_rows_start_column_df.columns = ['inflow']
                    selected_rows_start_column_df['building'] = selected_rows_start_column_df.index
                    selected_rows_start_column_df['day'] = interval.date()
                    selected_rows_start_column_df['interval'] = interval
                    # this is the df of rows of people coming into the building
                    inflow_data = pd.concat([inflow_data, selected_rows_start_column_df])

                # condition below is basically checking: t_i <= end_time < t_(i+1) i.e. end_time was in this
                # interval t_i
                selected_rows_end = building_stays_sample.loc[  # end_time happens in this interval
                    (building_stays_sample['end_time'] >= time_demarkations[i]) &
                    (building_stays_sample['end_time'] < time_demarkations[i + 1])]
                if selected_rows_end.shape[0] > 0:
                    selected_rows_end_column = selected_rows_end['building'].value_counts()
                    selected_rows_end_column_df = pd.DataFrame([selected_rows_end_column]).T
                    selected_rows_end_column_df.columns = ['outflow']
                    selected_rows_end_column_df['building'] = selected_rows_end_column_df.index
                    selected_rows_end_column_df['day'] = interval.date()
                    selected_rows_end_column_df['interval'] = interval
                    outflow_data = pd.concat([outflow_data,
                                              selected_rows_end_column_df])

                selected_rows_open = building_stays_sample.loc[  # start was before and end was after this interval
                    (building_stays_sample['start_time'] < time_demarkations[i]) &
                    (building_stays_sample['end_time'] > time_demarkations[i + 1])]
                """
                occupancy basically uses the concat of rows where people:
                - came in in this interval (selected_rows_start)
                - left in this interval (selected_rows_end)
                - were already present before and stayed after this interval (selected_rows_open)
                """
                occupancy_interval = pd.concat([
                    selected_rows_start,
                    selected_rows_end,
                    selected_rows_open
                ])

            if occupancy_interval.shape[0] > 0:
                occupancy_counts_column = occupancy_interval['building'].value_counts()
                occupancy_counts_column_df = pd.DataFrame([occupancy_counts_column]).T
                occupancy_counts_column_df.columns = ['distinct_individual_visits']
                occupancy_counts_column_df['building'] = occupancy_counts_column_df.index
                occupancy_counts_column_df['day'] = interval.date()
                occupancy_counts_column_df['interval'] = interval

                occupancy_data = pd.concat([occupancy_data, occupancy_counts_column_df])
        #     break #DEBUG
        # break #DEBUG

        # AGGREGATE hourly unique counts into mean and max per day
        def agg_func(df, stat, seed):
            df_agg = df[['building', 'day', stat]]
            df_agg = df_agg.groupby(['building', 'day']).agg(['mean', 'max'])

            df_agg.reset_index(inplace=True)
            df_agg.columns = ['building', 'day',
                              '{}_mean'.format(stat),
                              '{}_max'.format(stat),
                              ]
            df_agg['bootstrap'] = seed
            return df_agg

        # this calculates the mean and max OCCUPANCY per day (aggregated over the time intervals)
        occupancy_data_agg = agg_func(occupancy_data, 'distinct_individual_visits', seed)

        # this calculates the mean and max INFLOW per day (aggregated over the time intervals)
        inflow_data_agg = agg_func(inflow_data, 'inflow', seed)
        all_aggs = pd.merge(occupancy_data_agg, inflow_data_agg, how='left')

        # this calculates the mean and max OUTFLOW per day (aggregated over the time intervals)
        outflow_data_agg = agg_func(outflow_data, 'outflow', seed)
        all_aggs = pd.merge(all_aggs, outflow_data_agg, how='left')
        # NOTE: sometimes some columns in all_aggs might be NaN because

        # raw boostrap data to be used for intra-day trajectory
        all_data = pd.merge(occupancy_data, inflow_data, how='left')
        all_data = pd.merge(all_data, outflow_data, how='left')

        print('finished bootstrap:', seed, 'out of', self.n_bootstraps)

        return {'all_data': all_data,
                'all_aggs': all_aggs
                }

    def building_stats(self, building):

        if self.density == True:
            assert self.bootstrap_column == 'distinct_individual_visits', 'density can only be calculated for *occupancy* statistic'

        list_percentiles = []
        if self.building_data.is_valid_building_id(building) == False:
            print('invalid building:', building)
            return {}
        else:
            # definitions of buliding properties: http://web.mit.edu/ofms-space/www/textdocs/wbid.html#BREAKDOWN
            usable_area = self.building_data.building_properties(building).useable
            dates_dict = {}
            all_hf_day_dict = {}
            for day in self.day_list:
                data_this_building_this_day = self.all_boostrap_agg.loc[
                    (self.all_boostrap_agg['building'] == building)
                    &
                    (self.all_boostrap_agg['day'] == day)
                ]

                # calculating percentile for this building for this day
                distribution = data_this_building_this_day['{}_{}'.format(self.bootstrap_column, self.agg_type)]

                # checking that distribution has any data, OR that distribution does NOT contain just NaN
                if distribution.shape[0] > 0 or (not np.isnan(distribution).all()):
                    percentiles_vals = np.percentile(distribution, self.percentiles_list, interpolation='nearest')
                    if self.density == True:
                        percentiles_vals = percentiles_vals / int(usable_area)
                    list_percentiles.append(percentiles_vals)
                else:
                    percentiles_vals = [None for i in range(len(self.percentiles_list))]

                percentiles_dict = {}
                for i, val in enumerate(self.percentiles_list):
                    if percentiles_vals[i] is not None:
                        percentiles_dict[str(val) + ' percentile'] = sig_fig_formatted(percentiles_vals[i],
                                                                                       num_digits=3)
                    else:
                        # percentiles_dict[str(val) + ' percentile'] = None
                        percentiles_dict[str(val) + ' percentile'] = 0

                dates_dict[day.strftime('%Y-%m-%d')] = percentiles_dict
                # break #DEBUG

                # high_frequency data (since this does not depend on the mean of max aggregation, this gets set
                # upstream twice. but since dictionary keys are unique, it only gets outputted in the final json once
                high_freq_data = self.all_boostrap_data.loc[
                    (self.all_boostrap_data['building'] == building)
                    &
                    (self.all_boostrap_data['day'] == day)
                ]

                hf_timestamps = pd.date_range(  # generating 24 1-hour segments (assuming self.time_freq == '60min'
                    day,
                    periods=24,
                    freq=self.time_freq)

                hf_thisday_dict = {}
                for interval in hf_timestamps:
                    # selecting bootstraps over this single hour
                    hf_interval_data = high_freq_data[high_freq_data['interval'] == interval]
                    hf_distribution = hf_interval_data[self.bootstrap_column].dropna()

                    if hf_distribution.shape[0] > 0:
                        percentiles_vals = np.percentile(
                            hf_distribution, self.percentiles_list, interpolation='nearest')
                        if self.density == True:
                            percentiles_vals = percentiles_vals / int(usable_area)
                    else:
                        percentiles_vals = [0 for i in range(len(self.percentiles_list))]

                    percentiles_dict = {}
                    for i, val in enumerate(self.percentiles_list):
                        if percentiles_vals[i] is not None:
                            percentiles_dict[str(val) + ' percentile'] = sig_fig_formatted(percentiles_vals[i],
                                                                                           num_digits=3)
                        else:
                            percentiles_dict[str(val) + ' percentile'] = 0
                    all_hf_day_dict[str(interval)] = percentiles_dict

            # above was the calculation of percentiles per day. below is the calculation of percentils AVERAGED over all days
            all_percentiles_dict_without_timestamp = {}
            all_percentiles = np.vstack(list_percentiles)
            avg_percentiles = np.nanmean(all_percentiles, axis=0)
            for i, val in enumerate(self.percentiles_list):
                all_percentiles_dict_without_timestamp[str(
                    val) + ' percentile'] = sig_fig_formatted(avg_percentiles[i], num_digits=3)

            all_percentiles_dict = {
                self.first_day.strftime('%Y-%m-%d 00:00:00'): all_percentiles_dict_without_timestamp
            }

            if self.density == True:
                print('[{}] multiproc {} {} density'.format(building, self.bootstrap_column, self.agg_type))
            else:
                print('[{}] multiproc {} {}'.format(building, self.bootstrap_column, self.agg_type))

            return {'building': building,
                    'dates_dict': sort_dict(dates_dict),
                    'all_percentiles_dict': sort_dict(all_percentiles_dict),
                    'all_hf_day_dict': sort_dict(all_hf_day_dict)
                    }
