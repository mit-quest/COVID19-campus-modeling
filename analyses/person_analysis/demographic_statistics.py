import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

from pathlib import Path

from models.common.mit_buildings import MITBuildings
from models.common.to_precision import sig_fig_formatted
from analyses.common.analysis import Analysis, add_empty_buildings, sort_dict


def read_row(person):
    person_local = pd.DataFrame()

    person_local = person_local.append(
        {
            'home_zip': person['home_zip'],
            'mit_person_type': person['mit_person_type'],
            'age': person['age'],
            'commuting_type': person['commuting_type'].lower()
        }, ignore_index=True)

    return person_local


class ScenarioDemographicStatisticsAnalysis(Analysis):

    def run(self, input_samples: dict, input_analyses: dict, uuid_prefix: str) -> dict:
        np.random.seed(self.analysis_parameters['random_seed'])

        n_bootstraps = self.analysis_parameters['n_bootstraps']
        percentiles_list = self.analysis_parameters['percentiles']

        dates = input_samples['person']['dates']
        all_person_samples = input_samples['person']['samples']

        self.first_day = min(pd.to_datetime(dates))

        self.total_population_size = input_samples['person']['total_population_size']

        list_metrics = ['mean', 'max']

        print('cpu count on machine:', multiprocessing.cpu_count())
        cpu_count = multiprocessing.cpu_count()

        print('creating reading multiprocessing pool..')
        pool = multiprocessing.Pool(cpu_count)
        print('reading..')
        all_person_df = pd.concat(pool.map(read_row, all_person_samples))
        pool.close()
        pool.join()
        print('reading finished, closed pool')

        # print(all_person_df.head()) #DEBUG
        # print(all_person_df.shape) #DEBUG

        print('binning age')
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = ['1-10', '11-20', '21-30', '31-40', '41-50',
                      '51-60', '61-70', '71-80', '81-90', '91-100']

        all_person_df['age_bin'] = pd.cut(
            all_person_df['age'], bins=age_bins, labels=age_labels)

        self.all_person_df = all_person_df

        ######################## BOOTSTRAP ############
        pool = multiprocessing.Pool(cpu_count)
        print('created bootstrap multiprocessing pool..')
        output_list = pool.map(self.bootstrap, list(range(n_bootstraps)))
        pool.close()
        pool.join()
        print('bootstrapped finished, closed pool')

        output_list = list(zip(*output_list))
        all_commute_counts = pd.DataFrame(output_list[0])
        all_age_bin_counts = pd.DataFrame(output_list[1])
        all_home_zip_counts = pd.DataFrame(output_list[2])
        all_mit_person_type_counts = pd.DataFrame(output_list[3])

        list_boot_dataframes = [all_commute_counts, all_age_bin_counts,
                                all_home_zip_counts, all_mit_person_type_counts]
        label_statistics = ['commuting_type',
                            'age_bin', 'home_zip', 'mit_person_type']

        output_dict = {}

        statistics_dict = {}

        for i, stat in enumerate(label_statistics):
            df = list_boot_dataframes[i]
            stat_type = {}
            for column in df.columns:
                distribution = df[column].dropna()
                percentiles_vals = np.percentile(distribution, percentiles_list, interpolation='nearest')
                percentiles_dict = {}
                for i, val in enumerate(percentiles_list):
                    percentiles_dict[str(val) + ' percentile'] = sig_fig_formatted(percentiles_vals[i], int_only=True)
                stat_type[column] = percentiles_dict
            statistics_dict[stat] = {
                'full_simulation_time_window': {
                    'all_buildings': {
                        self.first_day.strftime('%Y-%m-%d 00:00:00'): stat_type
                    }
                }
            }

        path_to_plot, _ = os.path.split(__file__)
        path_to_plot = os.path.join(path_to_plot, 'plots')

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)

        statistics_dict = sort_dict(statistics_dict)

        statistics_dict['total_population_size'] = {
            'full_simulation_time_window': {
                'all_buildings': {
                    self.first_day.strftime('%Y-%m-%d 00:00:00'): {
                        'count': int(self.total_population_size)
                    }
                }
            }
        }

        output_dict['simulation'] = statistics_dict

        return(output_dict)

    def bootstrap(self, seed: int):
        '''
        this function takes in a random seed, and then selects a number <self.total_population_size> of
        people from the person samples to run a bootstrap. it then outputs the counts by age, zip,
        person type within this sample of people
        '''

        print('> bootstrap', seed)

        person_bootstrap_sample = self.all_person_df.sample(
            n=self.total_population_size,
            replace=True,
            random_state=seed
        )

        commute_counts = person_bootstrap_sample['commuting_type'].value_counts()
        age_bin_counts = person_bootstrap_sample['age_bin'].value_counts()
        home_zip_counts = person_bootstrap_sample['home_zip'].value_counts()
        mit_person_type_counts = person_bootstrap_sample['mit_person_type'].value_counts()

        return (commute_counts, age_bin_counts, home_zip_counts, mit_person_type_counts)
