import json
import os
import numpy as np
import pandas as pd
import random

from scipy.stats import norm
from pathlib import Path

from models.common.model import Model
from models.common.model_utils import day_of_week, normalize_to_one


def sum_to_x(n: int, x: float):
    if n > 1:
        values = [0.0, x] + list(np.random.uniform(low=0.0, high=x, size=n-1))
        values.sort()
        return [values[i+1] - values[i] for i in range(n)]
    if n == 1:
        return list(np.random.uniform(low=0.0, high=x, size=1))


def str_to_float_conversion(timestring: float):
    hh, mm = timestring.split(':')
    assert float(hh) < 24, '{} hour should not be more than 24'.format(timestring)
    return float(hh)+float(mm)/60.0


def float_to_str_conversion(timefloat: float):
    string = '{0:02.0f}:{1:02.0f}'.format(*divmod(timefloat * 60, 60))  # converts float time to HH:mm

    hh, mm = string.split(':')

    if mm == '60':
        hh = str(int(hh) + 1)
        mm = '00'
        string = hh+':'+mm

    if string == '24:00':
        string = '23:59'

    assert int(string.split(':')[0]) < 24, 'hh ('+str(timefloat)+') should be less than 24'

    return string


class Trajectory(Model):

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._parse_input_model_names()

        self.start_time_noise_sd = self.config['model_parameters']['start_time_noise_sd']
        self.end_time_noise_sd = self.config['model_parameters']['end_time_noise_sd']
        self.min_length_local_activity = self.config['model_parameters']['min_length_local_activity']

        self.go_into_work_pmf = pd.read_csv(self.config['model_parameters']['go_into_work_pmf'],
                                            index_col=0)['prob_go_into_work']

        self.go_into_class_pmf = pd.read_csv(self.config['model_parameters']['go_into_class_pmf'],
                                             index_col=0)['prob_go_into_class']

        schedule_cond_person_pmf_df_merged = pd.read_csv(self.config['model_parameters']['schedule_pmf'])
        schedule_cond_person_pmf_df_merged = schedule_cond_person_pmf_df_merged[
            ['wday', 'round_arrival', 'round_departure', 'local_activity_happened_marker', 'probability']]

        self.schedule_cond_person_pmf = {}
        """
        Note:
        If there was a local activity, local_activity_happened_marker will not be -1. So some trajectories have no
        local activity. i.e. -1 is how we indicate NO local activity in the schedule
        """
        for wday in range(7):
            schedule_cond_person_pmf_df_merged_by_day = schedule_cond_person_pmf_df_merged[
                schedule_cond_person_pmf_df_merged['wday'] == wday]
            schedule_cond_person_pmf_df_merged_by_day.reset_index(inplace=True)
            schedule_cond_person_pmf_df_merged_by_day = schedule_cond_person_pmf_df_merged_by_day[
                ['round_arrival', 'round_departure', 'local_activity_happened_marker', 'probability']]
            pairs_dict = {}
            for row_index in range(schedule_cond_person_pmf_df_merged_by_day.shape[0]):
                row = schedule_cond_person_pmf_df_merged_by_day.iloc[row_index]
                pairs_dict[(row['round_arrival'], row['local_activity_happened_marker'], row['round_departure'])] = row[
                    'probability']
            self.schedule_cond_person_pmf[wday] = pairs_dict

    def _parse_input_model_names(self) -> None:
        self.input_model_names = dict()
        for model_spec in self.config['input_models']:
            self.input_model_names[model_spec['model_type']] = model_spec['model_name']

    def sample(self, t_0: str, n_samples: int, dates: list, all_input_samples: dict) -> dict:
        """
        structure of "samples" key in input person model's json
        from json output is structure as such:
            person 1's sample trajectory with a list [...] for each day e.g:
                [
                    day 1 trajectory
                    day 2 trajectory
                    day 3 trajectory
                    .
                    .
                    .
                ]
            person 2's sample trajectory with a similar list of stays as above for each day
                [
                .
                .
                .
                ]
            person 3's sample trajectory with a similar list of stays as above for each day
            .
            .
            .
        """

        random.seed(self.model_parameters['random_seed'])
        np.random.seed(self.model_parameters['random_seed'])
        output_samples = list()
        for n in range(n_samples):
            inputs_sample = dict()
            for input_model_name, input_model_samples in all_input_samples.items():
                inputs_sample[input_model_name] = input_model_samples['samples'][n]

            output_samples.append(self.single_draw_from_trajectory_model(dates, inputs_sample))

        return dict(dates=dates,
                    samples=output_samples,
                    total_population_size=all_input_samples[self.input_model_names['person']]['total_population_size'])

    def single_draw_from_trajectory_model(self, dates: list, inputs_sample: dict) -> list:

        stays = []

        overnight_stays_queue = []

        for t_i, date in enumerate(dates):

            trajectory = []

            # don't let the person come into work before their first day returing
            if date < inputs_sample[self.input_model_names['person']]['first_day_returning']:
                stays.append([])
                continue

            building_open_fraction_this_day = inputs_sample[self.input_model_names['action']][t_i]

            et_overnight_stay = None

            day_name = day_of_week(date)['day_name']
            going_to_work_today = True
            # flip a biased coin to see if the person stays home this day
            if inputs_sample[self.input_model_names['person']]['mit_person_type'] == 'Undergraduate student':
                if np.random.rand() > self.go_into_class_pmf[day_name]:
                    # person does not go to work the today
                    going_to_work_today = False
            else:
                if np.random.rand() > self.go_into_work_pmf[day_name]:
                    # person does not go to work the next day
                    going_to_work_today = False

            # This block first checks that there are overnight stays. If they are not, and the person does not go to
            # work, it adds an empty list of stays [] then exits ('continue'). If there are overnight stays,
            # it adds the overnight stays to the current day's trajectories. if the person does not go to work, it exits
            # ('continue'). If they go to work, it exits the if statements and continues down the rest of the code
            if len(overnight_stays_queue) == 0:
                if going_to_work_today == False:
                    stays.append([])
                    continue
            else:
                trajectory = []
                for overnight_stay in overnight_stays_queue:
                    trajectory.append(overnight_stay)
                    if 'departure_time' in overnight_stay:
                        et_str = overnight_stay['departure_time']
                        et_hh, et_mm = et_str.split(':')
                        # et_overnight_stay is the end time of the overnight trajectories so that if there is a new
                        # trajectory today, it will be verified that this new trajectory's st > et_overnight_stay
                        et_overnight_stay = float(et_hh) + float(et_mm)/60
                if going_to_work_today == False:
                    stays.append(trajectory)
                    overnight_stays_queue = []
                    continue
                else:
                    overnight_stays_queue = []

            buildings = inputs_sample[self.input_model_names['person']]['building_and_core_access'][0]
            random.shuffle(buildings)

            num_buildings = len(buildings)
            assert num_buildings > 0

            arrival_departure_pmf_dict = self.schedule_cond_person_pmf[day_of_week(date)['day_index']]
            arrival_departure_value_indices = range(len(arrival_departure_pmf_dict.keys()))
            arrival_departure_value_prob = normalize_to_one(list(arrival_departure_pmf_dict.values()))

            """
            this code block creates the start and end time of the stay
            """
            if et_overnight_stay:  # checking that it is not null, i.e. there was an overnight trajectory
                st = float('-inf')  # to make sure st starts with a value less than et so pointer gets into the loop
                # to generate a new st
                counter = 0
                while st < et_overnight_stay:
                    index_choice = np.random.choice(arrival_departure_value_indices, p=arrival_departure_value_prob)
                    (st, local_activity_happened, et) = list(arrival_departure_pmf_dict.keys())[index_choice]
                    counter = counter + 1
                    if counter > 1000:
                        # TODO find a better way that does not violate probability distribution thru this hack
                        # we add 0.00000000000001 so that st still happens after the last stay, and so that it passes
                        # the test down (assert st+time_intervals[-1] != et)
                        st = et_overnight_stay + 0.00000000000001
                        break

            else:  # there was no overnight stay
                index_choice = np.random.choice(arrival_departure_value_indices, p=arrival_departure_value_prob)
                (st, local_activity_happened, et) = list(arrival_departure_pmf_dict.keys())[index_choice]
                st = st + np.random.uniform(0, self.start_time_noise_sd)

            # the distribution above give us start time and end time as round numbers. within the hour,
            # we pick the actual startime from a random uniform distribution
            et = et + np.random.uniform(0, self.end_time_noise_sd)
            # TODO what if the start time + noise is after endtime when st and et are very close apart.

            if num_buildings > 1:
                if et < st:
                    # this means that et is after midnight, so we add 24 to it to be able to calculate difference in
                    # times between the two (otherwise we get negative time intervals)
                    et = et + 24
                time_intervals = sum_to_x(num_buildings - 1, et-st)
                time_intervals.sort()
                time_intervals.insert(0, 0)
                assert st+time_intervals[-1] != et, 'no time between the last stay end, and actual end time'

            else:  # i.e. num_buildings == 1 (since we assert num_buildings > 0 earlier)
                if et < st:  # just one building
                    # this means that et is after midnight
                    et = et + 24
                time_intervals = [0]

            assert et - st < 24, 'departure time {} should be within 24 hours of arrival time {}'.format(et, st)

            trajectory.append(
                {
                    'stay_type': 'commute',
                    'commute_type': inputs_sample[self.input_model_names['person']]['commuting_type'],
                    'start_zip_code': inputs_sample[self.input_model_names['person']]['home_zip'],
                    'arrival_time': float_to_str_conversion(st)
                }
            )

            """
            this code block creates the stays accounting for the complications of overnight stays
            """
            for i, t in enumerate(time_intervals):
                try:
                    stay_end_time_float = st+time_intervals[i+1]
                except IndexError:  # reached end of array
                    stay_end_time_float = et

                # this block is messy but it is checking for the following possibilities:
                """

                                        go to work today
                                              |
                               NO+------------+-------------+YES
                                 |                          |
                      overnight trajectory         NO overnight trajectory
                      from last night              from last+night
                             |                              |
                    NO+------+-------+YES           NO+-----+---------+YES
                      |              |                |               |
                empty traj []    overnight        today's traj    today's traj
                                 trajectory       only                plus
                                                                  overnight traj

                """

                attempted_building = buildings[i]['building']
                # if this building is not currently open, then the person is stuck outside
                if np.random.rand() < building_open_fraction_this_day[attempted_building]:
                    this_stay_building = attempted_building
                else:
                    this_stay_building = 'outside'

                if stay_end_time_float > 24:
                    if st+time_intervals[i] < 24:  # start time of this stay is before midnight
                        # if the stay started before midnight and ended after midnight, we add the before part
                        # in this day's trajectory, and the after part in the next day's trajectory
                        stay = {
                            'stay_type': 'on_campus_inside',
                            'building': this_stay_building,
                            'start_time': float_to_str_conversion(st + time_intervals[i]),
                            'end_time': '23:59'
                        }
                        trajectory.append(stay)

                        # tomorrow's stay section in tomorrow's overnight_stays_queue
                        stay = {
                            'stay_type': 'on_campus_inside',
                            'building': this_stay_building,
                            'start_time': '00:00',
                            'end_time': float_to_str_conversion(stay_end_time_float-24)
                        }
                        overnight_stays_queue.append(stay)

                        if i == len(time_intervals) - 1:  # this is the last stay so we add commute
                            overnight_stays_queue.append({
                                'stay_type': 'commute',
                                'commute_type': inputs_sample[self.input_model_names['person']]['commuting_type'],
                                'end_zip_code': inputs_sample[self.input_model_names['person']]['home_zip'],
                                'departure_time': float_to_str_conversion(et-24)
                            }
                            )
                    else:  # this is the case where the start and end of this stay is after midnight, so we dont have
                        # to break the stay
                        stay = {
                            'stay_type': 'on_campus_inside',
                            'building': this_stay_building,
                            'start_time': float_to_str_conversion(st + time_intervals[i] - 24),
                            'end_time': float_to_str_conversion(stay_end_time_float - 24)
                        }
                        overnight_stays_queue.append(stay)

                        if i == len(time_intervals) - 1:  # this is the last stay so we add commute
                            overnight_stays_queue.append({
                                'stay_type': 'commute',
                                'commute_type': inputs_sample[self.input_model_names['person']]['commuting_type'],
                                'end_zip_code': inputs_sample[self.input_model_names['person']]['home_zip'],
                                'departure_time': float_to_str_conversion(et-24)
                            }
                            )

                else:  # stay starts and ends before midnight i.e. stay_end_time_float > 24
                    stay = {
                        'stay_type': 'on_campus_inside',
                        'building': this_stay_building,
                        'start_time': float_to_str_conversion(st+time_intervals[i]),
                        'end_time': float_to_str_conversion(stay_end_time_float)
                    }
                    trajectory.append(stay)

                    if i == len(time_intervals) - 1:  # this is the last stay so we add commute
                        trajectory.append({
                            'stay_type': 'commute',
                            'commute_type': inputs_sample[self.input_model_names['person']]['commuting_type'],
                            'end_zip_code': inputs_sample[self.input_model_names['person']]['home_zip'],
                            'departure_time': float_to_str_conversion(et)
                        }
                        )

            """
            If there was a local activity, local_activity_happened_marker will not be -1. So some trajectories have no
            local activity. i.e. -1 is how we indicate NO local activity in the schedule
            """
            if local_activity_happened != -1:
                trajectory = self.add_local_activity(trajectory, local_activity_happened)

            stays.append(trajectory)

        return stays

    def add_local_activity(self, trajectory_to_update: list, local_activity_happened: float):
        """
            # The way this is currently done is that a triple of `arrival_time` (time coming into campus), `local_activity_happened_marker` (start time of first local activity, rounded to the nearest hour), and `departure_time` (time leaving campus) are sampled from a joint distribution provided by Bernardo from Cuebiq data (`models/trajectory/trajectory_covid_access_ocr_rampup/data/arrival_outing_departure_june.json`).
            # Trajectories are generated as before (including overnight) and right before returning a person's sample trajectory, if this sampled triple has a local activity (if not, `local_activity_happened_marker` i.e.  `local_activity_happened` would be equal to -1)  it is pushed through the function `add_local_activity` which scans for the stay where the start time of the local activity (`local_activity_happened`) falls into. This stay is then split into two parts with a new local activity stay sandwiched in between. Note that the stay has a `location` key which can later be used to specify a specific local destination (e.g. Chipotle) but is left `null` for now due to lack of data.
        """

        updated_trajectory = []
        for stay in trajectory_to_update:
            if stay['stay_type'] == 'commute':
                updated_trajectory.append(stay)
                continue  # skip
            else:
                st = str_to_float_conversion(stay['start_time'])
                et = str_to_float_conversion(stay['end_time'])

                if (local_activity_happened > st) and (local_activity_happened < et):  # this stay needs to be split
                    updated_trajectory.append(  # adding first part of stay that is before local activity
                        {
                            'stay_type': stay['stay_type'],
                            'building': stay['building'],
                            'start_time': float_to_str_conversion(st),
                            'end_time': float_to_str_conversion(local_activity_happened)
                        }
                    )

                    # generating local activity end time, local_activity_end_time
                    # TODO sample local_activity_end_time from data instead of randomly sampling
                    local_activity_end_time = local_activity_happened + np.random.uniform(0, self.end_time_noise_sd)
                    counter = 0
                    # local activity end time cannot be after stay end, and making sure outing is at least 15min (
                    # when self.min_length_local_activity = 0.25 hours)
                    while local_activity_end_time > et or (local_activity_end_time-local_activity_happened) < self.min_length_local_activity:
                        local_activity_end_time = local_activity_happened + \
                            np.random.uniform(0, self.end_time_noise_sd)
                        counter = counter + 1
                        if counter > 1000:
                            # if we can't get randomly generated values, we just split the stay in the middle
                            local_activity_end_time = local_activity_happened + (
                                et-local_activity_happened)/2.0
                            break

                    assert local_activity_end_time < et, 'local activity end time {} cannot be after stay end {' \
                                                         '}'.format(local_activity_end_time, end_time)

                    updated_trajectory.append(  # local activity
                        {
                            'stay_type': 'local_activity',
                            # we don't have a location for the outing yet, but adding it as a placeholder
                            'location': None,
                            'start_time': float_to_str_conversion(local_activity_happened),
                            'end_time': float_to_str_conversion(local_activity_end_time)
                        }
                    )

                    updated_trajectory.append(  # adding last part of stay that is after local activity
                        {
                            'stay_type': stay['stay_type'],
                            'building': stay['building'],
                            'start_time': float_to_str_conversion(local_activity_end_time),
                            'end_time': float_to_str_conversion(et)
                        }
                    )
                else:  # this stay did not contain the local activity
                    updated_trajectory.append(stay)

        return updated_trajectory
