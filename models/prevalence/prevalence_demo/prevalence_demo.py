import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import odeint

from models.common.model import Model
"""
active cases as of 2022-03-22 as per https://www.worldometers.info/coronavirus/country/us/: 7198983
population of US: 328200000
prevalence: 7198983 / 328200000 = 0.0022
"""


class PrevalenceDemo(Model):
    def sample(self, t_0: str, n_samples: int, dates: list, all_input_samples: dict) -> dict:
        np.random.seed(self.model_parameters['random_seed'])
        dir_path = os.path.dirname(os.path.abspath(__file__))
        common_path = os.path.join(dir_path, '..', '..', 'common')

        with open(Path(self.model_parameters['zip_code_fpath']).as_posix()) as f:
            self.ZIP_CODES = f.read().splitlines()

        output_samples = list()
        for n in range(n_samples):
            sample_dict = {}

            # below is where you would build a SEIR-type model to predict future infection levels from past data
            # for now this is just using a national level number with some uniform noise
            # or you can get the daily forecast from the CDC: https://www.cdc.gov/coronavirus/2019-ncov/covid-data/forecasting-us.html
            for i in range(len(dates)):
                zip_dict = {}
                for zip_code in self.ZIP_CODES:
                    zip_dict[zip_code] = {
                        'prevalence_uncontained_infections': 0.0022 + np.random.rand()/1000
                    }
                sample_dict[i] = zip_dict

            output_samples.append(sample_dict)

        return dict(dates=dates, samples=output_samples)
