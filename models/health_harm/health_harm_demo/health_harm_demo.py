import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from models.common.model import Model

HEALTH_HARMS = ('infection', 'sickness',
                'hospitalization', 'ICU care', 'death')

# https://www.hopkinsmedicine.org/health/conditions-and-diseases/coronavirus/diagnosed-with-covid-19-what-to-expect
# Those with mild cases of COVID-19 appear to recover within one to two weeks. For severe cases, recovery may take six weeks or more.

# https://www.npr.org/sections/health-shots/2020/03/31/824155179/cdc-director-on-models-for-the-months-to-come-this-virus-is-going-to-be-with-us
# 25% of people do not show get symptoms

# https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html
# Symptoms may appear 2-14 days after exposure to the virus.

# approximated from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm
# top level key is assumed to be max age range that the stats apply to
prob_health_outcome_by_age_data = {
    19: {
        'hospitalization': 0.02,
        'ICU care': 0.0,
        'death': 0.0
    },
    44: {
        'hospitalization': 0.16,
        'ICU care': 0.03,
        'death': 0.002
    },
    54: {
        'hospitalization': 0.25,
        'ICU care': 0.08,
        'death': 0.007
    },
    64: {
        'hospitalization': 0.25,
        'ICU care': 0.1,
        'death': 0.02
    },
    74: {
        'hospitalization': 0.35,
        'ICU care': 0.13,
        'death': 0.035
    },
    84: {
        'hospitalization': 0.44,
        'ICU care': 0.2,
        'death': 0.07
    },
    1000: {
        'hospitalization': 0.5,
        'ICU care': 0.18,
        'death': 0.19
    }
}


def pmf_over_health_outcome(age):
    key_i = np.where(
        np.array(list(prob_health_outcome_by_age_data.keys())) > age)[0][0]
    return prob_health_outcome_by_age_data[list(prob_health_outcome_by_age_data.keys())[key_i]]


class HealthHarmDemo(Model):
    def sample(self,  t_0: str, n_samples: int, dates: list, all_input_samples: dict) -> dict:
        self.harms = HEALTH_HARMS
        np.random.seed(self.model_parameters['random_seed'])

        output_samples = list()
        for n in range(n_samples):
            inputs_sample = dict()
            # Get all input model values for sample n
            for input_model_name, input_model_samples in all_input_samples.items():
                inputs_sample[input_model_name] = input_model_samples['samples'][n]
            output_samples.append(
                self.single_draw_from_model(dates, inputs_sample))
        return dict(dates=dates, samples=output_samples)

    def single_draw_from_model(self, dates: list, inputs_sample: dict) -> list:
        harms_df = self.sample_harms_df(inputs_sample['infection_demo'],
                                        inputs_sample['person']['age'])
        return list((harms_df.transpose().to_dict()).values())

    def sample_harms_df(self, exposure_array, age):
        # see model_data.harms for approximate statistics that were used here
        harms_df = pd.DataFrame(False, index=range(
            len(exposure_array)), columns=self.harms)
        if not np.array(exposure_array).any():
            # was not exposed to SARS-CoV-2
            return harms_df
        symptoms, hospitalization, icu_admission, death = False, False, False, False
        if np.random.rand() <= 0.75:
            symptoms = True
            outcomes_pmf = pmf_over_health_outcome(age)
            if np.random.rand() <= outcomes_pmf['hospitalization']:
                hospitalization = True
                if np.random.rand() <= outcomes_pmf['ICU care']:
                    icu_admission = True
                    if np.random.rand() <= outcomes_pmf['death']:
                        death = True
        first_exposure = np.where(np.array(exposure_array))[0][0]
        if not symptoms:
            n_days_infected = np.random.randint(2, 7*2+1)
            harms_df['infection'][first_exposure:first_exposure +
                                  n_days_infected] = True
            return harms_df
        if not hospitalization:
            n_days_infected = np.random.randint(7, 7*3+1)
            harms_df['infection'][first_exposure:first_exposure +
                                  n_days_infected] = True
            harms_df['sickness'][first_exposure +
                                 2:first_exposure+n_days_infected-2] = True
            return harms_df
        n_days_infected = np.random.randint(7*3, 7*6+1)
        harms_df['infection'][first_exposure:first_exposure +
                              n_days_infected] = True
        harms_df['sickness'][first_exposure +
                             2:first_exposure + n_days_infected - 2] = True
        harms_df['hospitalization'][first_exposure +
                                    5:first_exposure + n_days_infected - 5] = True
        if not icu_admission:
            return harms_df
        harms_df['ICU care'][first_exposure +
                             10:first_exposure + n_days_infected - 10] = True
        if not death:
            return harms_df
        harms_df[:][first_exposure + 14+1:-1] = False
        harms_df['death'][first_exposure + 14:-1] = True
        return harms_df


class PlaceholderInBuildingHealthHarmModel(HealthHarmDemo):
    def single_draw_from_model(self, dates: list, inputs_sample: dict) -> list:
        harms_df = self.sample_harms_df(inputs_sample['in-building-infection-model'],
                                        inputs_sample['placeholder-mit-person-model']['age'])
        return list((harms_df.transpose().to_dict()).values())
