import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


from models.common.model import Model


def type_of_day(date):
    if pd.to_datetime(date).weekday() <= 4:
        return 'weekday'
    return 'weekend'


class InfectionDemo(Model):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._parse_input_model_names()

    def sample(self, t_0: str, n_samples: int, dates: list, all_input_samples: dict) -> dict:
        random.seed(self.model_parameters['random_seed'])
        np.random.seed(self.model_parameters['random_seed'])
        output_samples = list()
        for n in range(n_samples):
            inputs_sample = dict()
            for input_model_name, input_model_samples in all_input_samples.items():
                inputs_sample[input_model_name] = input_model_samples['samples'][n]

            trajectory_sample = inputs_sample[self.input_model_names['trajectory']]
            person_sample = inputs_sample[self.input_model_names['person']]
            prevalence_sample = inputs_sample[self.input_model_names['prevalence']]

            interaction_sample = self.single_draw_from_interaction_model(trajectory_sample, person_sample, dates)
            # trajectory_sample['placeholder-zip-code-commute-mobility-model'] = self.single_draw_from_mobility_model("commute", dates, inputs_sample)
            output_samples.append(
                self.single_draw_from_infection_model(
                    dates, inputs_sample,
                    interaction_sample,
                    prevalence_sample
                )
            )

        return dict(dates=dates, samples=output_samples)

    def single_draw_from_interaction_model(self, trajectory_sample: dict, person_sample: dict, dates: list) -> list:
        interactions = []

        zip_codes = pd.read_csv(self.config['model_parameters']['zip_codes_data'],
                                dtype={'zipcode': str, 'distance': float})
        distance_per_hour = self.config['model_parameters']['distance_per_hour']  # KM because it's the superior unit
        commuting_contacts_per_hour = pd.read_csv(self.config['model_parameters']['commuting_contacts_per_hour'])
        FMT = '%H:%M'

        for t_i, date in enumerate(dates):

            # this person did not go to work, trajectory sample is empty
            if len(trajectory_sample[t_i]) == 0:
                # interactions.append( [] )
                hours_at_work = 0
            else:
                # print(trajectory_sample[t_i])

                for s_i, stay in enumerate(trajectory_sample[t_i]):

                    if s_i == 0:
                        # this is a commute in stay
                        # print('s_i', s_i)
                        try:
                            this_person_zipcode = stay['start_zip_code']
                            this_person_commute_type = stay['commute_type']
                            this_person_arrival_time = stay['arrival_time']

                            distance_to_commute = zip_codes[zip_codes['zipcode']
                                                            == this_person_zipcode]['distance'].values[0]

                            time_commuting = distance_to_commute/distance_per_hour + 1

                            commuting_interactions_max = int(
                                time_commuting *
                                commuting_contacts_per_hour.loc[
                                    commuting_contacts_per_hour['commuting_type'] == this_person_commute_type, 'contacts_per_hour'].values[0]
                            )
                        except:
                            # if we have an overnight trajectory, there is no commute for the first stay the next day
                            commuting_interactions_max = 0
                            this_person_arrival_time = stay['start_time']
                            this_person_zipcode = '02139'

                    # last stay
                    if s_i == len(trajectory_sample[t_i]) - 1:
                        # if we have an overnight trajectory, there is no departure time for the first part of the stay
                        try:
                            this_person_end_time = stay['departure_time']
                        except:
                            this_person_end_time = stay['end_time']

                # hours_at_work = inputs_sample['person_joint_sampling_covid_access_ocr_rampup']['hours_at_work'][type_of_day(date)]
                seconds_at_work = datetime.strptime(this_person_end_time, FMT) - \
                    datetime.strptime(this_person_arrival_time, FMT)
                hours_at_work = seconds_at_work.seconds/60/60
                interactions_per_hour = 10
                work_interactions_max = int(hours_at_work*interactions_per_hour)

            if hours_at_work > 0:  # hooman went to work
                # print(this_person_zipcode)
                todays_interaction = dict(
                    commute_interactions=self.commute_interactions(
                        date, commuting_interactions_max, this_person_zipcode),
                    building_interaction=self.building_interaction(date, work_interactions_max))
            else:
                todays_interaction = dict(commute_interactions=dict(), building_interaction=dict())

            interactions.append(todays_interaction)

        return interactions

    def commute_interactions(self, date: str, commuting_interactions_max: int, this_person_zipcode: str) -> dict:

        interactions = {
            this_person_zipcode: np.random.randint(
                0, commuting_interactions_max) if commuting_interactions_max > 0 else 0
        }
        return interactions

    def building_interaction(self, date: str, work_interactions_max: int) -> dict:
        shared_space_use = {
            '02139': np.random.randint(0, work_interactions_max) if work_interactions_max > 0 else 0
        }
        return shared_space_use

    def single_draw_from_infection_model(self, dates: list, inputs_sample: dict, interaction_sample: dict, prevalence_sample: dict) -> list:
        infections = list()
        for t_i, date in enumerate(dates):
            infected = False
            for interaction_type in interaction_sample:

                if len(interaction_type['commute_interactions']) > 0:
                    # in case person commutes thru many zipcodes
                    for zip, n_interactions in interaction_type['commute_interactions'].items():
                        # population_zip_code = 1000
                        # print(t_i, zip)
                        prevalence_zip_code = prevalence_sample[str(t_i)][zip]['prevalence_uncontained_infections']
                        p_infect = (self.model_parameters['prob_interactions_infect']
                                    * n_interactions * prevalence_zip_code)
                        if np.random.rand() < p_infect:
                            infected = True

                if len(interaction_type['building_interaction']) > 0:
                    for zip, n_shared in interaction_type['building_interaction'].items():
                        # population_zip_code = 1000
                        prevalence_zip_code = prevalence_sample[str(t_i)][zip]['prevalence_uncontained_infections']
                        p_infect = (self.model_parameters['prob_shared_space_infect'] * n_shared *
                                    prevalence_zip_code)
                        if np.random.rand() < p_infect:
                            infected = True

            infections.append(infected)
        return infections

    def _parse_input_model_names(self) -> None:
        self.input_model_names = dict()
        for model_spec in self.config['input_models']:
            self.input_model_names[model_spec['model_type']] = model_spec['model_name']
