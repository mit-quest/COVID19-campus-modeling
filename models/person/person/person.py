import json
import os
import numpy as np
import pandas as pd
import random

from pathlib import Path
from scipy.stats import norm

from models.common.model import Model
from models.common.mit_buildings import MITBuildings


def normalize_to_one(counts_list):
    return [pi/sum(counts_list) for pi in list(counts_list)]


class Person(Model):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._parse_input_model_names()

        self.building_data = MITBuildings()

        np.random.seed(self.model_parameters['random_seed'])
        random.seed(self.model_parameters['random_seed'])

        mit_buildings = MITBuildings().known_building_ids()
        self.mit_building_pmf = {
            b: 1.0/len(mit_buildings) for b in sorted(mit_buildings)}

        with open(self.config['model_parameters']['mit_person_age_pdf'], 'r') as fp:
            mit_person_age_pdf_data = pd.read_csv(fp)
            mit_person_age_pdf_data.index = ['mean', 'std']

        self.mit_person_age_pdf = {}
        for person_type in mit_person_age_pdf_data.keys():
            self.mit_person_age_pdf[person_type] = norm(mit_person_age_pdf_data[person_type]['mean'],
                                                        mit_person_age_pdf_data[person_type]['std'])

        with open(self.config['model_parameters']['commute_cond_person_pmf'], 'r') as fp:
            self.mit_commuting_pmf = json.load(fp)

        self.zip_cond_commute_list = pd.read_csv(
            self.config['model_parameters']['zip_to_commute_mapping'], dtype=str)

        self.building_assignment_pmf = pd.read_csv(self.config['model_parameters']['building_assignment_pmf'])

        # sanitizing buildings
        for blg in self.building_assignment_pmf['building']:
            if not self.building_data.is_valid_building_id(blg):
                # dropping rows where invalid blg are
                self.building_assignment_pmf = self.building_assignment_pmf[
                    self.building_assignment_pmf['building'] != blg]
            else:
                # replace ill-formatted (but valid) buildings by their approved names e.g. 14N -> 14
                if blg != self.building_data.pretty_num(blg):
                    self.building_assignment_pmf.replace(blg,
                                                         self.building_data.pretty_num(blg),
                                                         inplace=True)

        # normalize self.building_assignment_pmf probablities
        self.building_assignment_pmf['prob_of_assignment'] = self.building_assignment_pmf['probability']/sum(
            self.building_assignment_pmf['probability'])

        self.mit_person_type_pmf = pd.read_csv(self.config['model_parameters']['mit_person_type_pmf'], dtype=float).T[0]
        self.total_population_size = None
        self.person_type_first_day_returning_pmf = None

    def _parse_input_model_names(self) -> None:
        self.input_model_names = dict()
        for model_spec in self.config['input_models']:
            self.input_model_names[model_spec['model_type']] = model_spec['model_name']

    def set_population_size_and_first_day_pmf(self, dates: list) -> None:
        # The total population is a function of the date range so some of the attributes need to be created here
        # and not in __init__
        # TODO add in dorms and on/off campus

        # to ensure the simulation starts off with a population, the first date must not be before the first
        # injection of people
        assert min(self.config['model_parameters']['people_to_inject'].keys()) <= dates[0]

        desired_population = pd.DataFrame(
            index=pd.date_range(min(self.config['model_parameters']['people_to_inject'].keys()), dates[-1]),
            columns=['Undergraduate student'] + list(self.mit_person_type_pmf.index),
            data=0)

        for injection_date in sorted(self.config['model_parameters']['people_to_inject'].keys()):
            if injection_date not in desired_population.index:
                break
            injections = self.config['model_parameters']['people_to_inject'][injection_date]
            for injection in injections:
                desired_population[injection['type']][injection_date:] += injection['amount']

        for daily_increase in self.config['model_parameters']['population_daily_increase']:
            for day in desired_population.index[1:]:
                desired_population.loc[day:, daily_increase['type']] += daily_increase['amount']

        """
        this creates a pd dataframe of probabily of returning for each person type and date
        e.g.
                       person_type       date      prob
        0    Undergraduate student 2020-06-15  0.049319 -> prob(person returns | undergrad, june 15) = 0.049
        1    Undergraduate student 2020-06-16  0.000000
        2    Undergraduate student 2020-06-17  0.000000
        3    Undergraduate student 2020-06-18  0.000000
        4    Undergraduate student 2020-06-19  0.000000
        """
        self.person_type_first_day_returning_pmf = desired_population.diff()
        self.person_type_first_day_returning_pmf.iloc[0] = desired_population.iloc[0]
        self.person_type_first_day_returning_pmf = (self.person_type_first_day_returning_pmf
                                                    / self.person_type_first_day_returning_pmf.sum().sum())
        self.person_type_first_day_returning_pmf = self.person_type_first_day_returning_pmf.unstack().reset_index(
            name='prob').rename(columns={'level_0': 'person_type', 'level_1': 'date'})
        self.total_population_size = int(desired_population.loc[dates[-1]].sum())

    def sample(self, t_0: str, n_samples: int, dates: list, all_input_samples: dict) -> dict:

        self.set_population_size_and_first_day_pmf(dates)

        output_samples = list()
        for n in range(n_samples):
            inputs_sample = dict()
            # Get all input model values for sample n
            for input_model_name, input_model_samples in all_input_samples.items():
                inputs_sample[input_model_name] = input_model_samples['samples'][n]
            output_samples.append(
                self.single_draw_from_model(dates, inputs_sample))

        return dict(dates=dates, samples=output_samples, total_population_size=self.total_population_size)

    def single_draw_from_model(self, dates: list, inputs_sample: dict) -> dict:

        ind = np.random.choice(self.person_type_first_day_returning_pmf.index,
                               p=self.person_type_first_day_returning_pmf['prob'])
        mit_person_type = self.person_type_first_day_returning_pmf['person_type'][ind]
        first_day_returning = str(self.person_type_first_day_returning_pmf['date'][ind].date())

        commuting = np.random.choice(list(self.mit_commuting_pmf.keys()), p=list(self.mit_commuting_pmf.values()))
        commuting = commuting.lower()

        zipcode_cond_commute = self.zip_cond_commute_list[commuting]
        zipcode_cond_commute = [i for i in zipcode_cond_commute.values if not pd.isnull(i)]

        home_zip = random.choice(zipcode_cond_commute)

        age = int(round(self.mit_person_age_pdf[mit_person_type].rvs()))

        # TODO add graduate students who live on campus (02139 zip codes and walk)
        # flip a biased coin for each building, to see if the person is assigned to that building
        if mit_person_type == 'Undergraduate student':
            n = self.config['model_parameters']['n_different_classroom_buildings_per_student']
            student_buildings = self.config['model_parameters']['student_classroom_buildings']
            list_buildings = list(np.random.choice(student_buildings, n, replace=False))
            home_zip = '02139'
            commuting = 'walk'
        elif mit_person_type == 'Service staff':
            num_buildings = self.config['model_parameters']['service_staff_building_visited']
            list_buildings = list(np.random.choice(self.building_assignment_pmf['building'],
                                                   size=num_buildings,
                                                   p=self.building_assignment_pmf['prob_of_assignment'],
                                                   replace=False))
        else:
            # generate random vector of probabilities for each row (i.e. building in self.building_assignment_pmf)
            individual_building_probabilities = np.random.rand(self.building_assignment_pmf.shape[0])
            # compare generated probability with self.building_assignment_pmf row (i.e. building) probability to
            # select (True or False) each row (i.e. building)
            individual_building_selection = individual_building_probabilities < self.building_assignment_pmf[
                'prob_of_assignment']
            list_buildings = self.building_assignment_pmf[individual_building_selection]['building'].to_list()
            if len(list_buildings) == 0:
                # if no buildings were sampled (happens rarely),
                # to ensure they at least have a building uniformly sample from all buildings
                list_buildings.append(np.random.choice(self.building_assignment_pmf['building'].to_list()))

        # sanitizing buildings to remove those that are not valid names as per MITBuildings class
        # we have to do it again because some of the buildings comes from the undergrad class and not from self.building_assignment_pmf
        list_buildings_sanitized = [self.building_data.pretty_num(b) for b in list_buildings if
                                    self.building_data.is_valid_building_id(b)]

        list_buildings_dicts = [{'building': b} for b in list_buildings_sanitized]

        # these should always be at least one building
        assert len(list_buildings_dicts) > 0, 'building list before sanitization: {}'.format(list_buildings)

        # list_cores = []

        person = dict(
            mit_person_type=mit_person_type,
            home_zip=home_zip,
            commuting_type=commuting,
            age=age,
            building_and_core_access=[list_buildings_dicts],
            first_day_returning=first_day_returning
        )

        return person
