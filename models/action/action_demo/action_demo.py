import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from models.common.model import Model
from models.common.mit_buildings import MITBuildings


class ActionDemo(Model):
    def sample(self,  t_0: str, n_samples: int, dates: list, input_samples: dict) -> dict:
        self.mit_buildings = MITBuildings()

        np.random.seed(self.model_parameters['random_seed'])
        self.actions = list(self.reopen_schedule(dates).transpose().to_dict().values())
        return dict(dates=dates, samples=[self.actions for _ in range(n_samples)])

    def reopen_schedule(self, dates):
        reopen_df = pd.DataFrame(
            0.0, index=dates, columns=self.mit_buildings.known_building_ids())

        reopened_buildings = []
        for b in self.mit_buildings.known_building_ids():
            if np.random.rand() > (self.model_parameters['percent_campus_to_reopen'] / 100.0):
                # this building won't reopen
                continue
            reopened_buildings.append(b)
            final_reopen = round(np.random.rand(), 2)
            if np.random.rand() > (self.model_parameters['percent_buildings_that_were_closed'] / 100.0):
                # this building stayed open
                reopen_start_i = 0
            else:
                reopen_start_i = np.random.choice(range(len(dates)))
            num_weeks = int(np.ceil((len(dates) - reopen_start_i) / 7))
            for week_i in range(1, num_weeks+1):
                # Josh what is going on here? I think:
                # for the time interval reopen_start_i+7*(week_i-1) to
                # reopen_start_i+(7*week_i), we set the opening fraction
                # of building b to be round(week_i * final_reopen / num_weeks, 2)
                reopen_df[b][reopen_start_i+7*(week_i-1):reopen_start_i+(7*week_i)] = \
                    round(week_i * final_reopen / num_weeks, 2)

        # reopen_df = reopen_df[reopened_buildings]
        return reopen_df
