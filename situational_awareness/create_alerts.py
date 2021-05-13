import json
import os
import re

OCCUPANCY_ALERT_THRESHOLD = 0.75


def read_json(fp):
    file_object = open(fp, 'r')
    dict = json.load(file_object)
    return dict


def calculate_alert_dict(occupancy_dict, building_counts_dict):
    buildings = occupancy_dict['simulation']['occupancy']['per_hour'].keys()
    alert_dict = {}
    not_in_counts = []

    for building in buildings:
        if building not in building_counts_dict:
            not_in_counts.append(building)
        else:
            threshold = OCCUPANCY_ALERT_THRESHOLD
            totalAssigned = building_counts_dict[building]
            alert_dict[building] = {}
            date_times = occupancy_dict['simulation']['occupancy']['per_hour'][building].keys()

            for date_time in date_times:
                estimatedOccupancy = occupancy_dict['simulation']['occupancy']['per_hour'][building][date_time][
                    '50 percentile']
                alert_dict[building][date_time] = {
                    'estimated_occupancy': estimatedOccupancy,
                    'total_unique_people_assigned': totalAssigned,
                    'alert_status': estimatedOccupancy > (threshold * totalAssigned)
                }

    top_level_alert_dict = {'simulation': {
        'occupancy_alerts': {
            'parameters': {
                'threshold': OCCUPANCY_ALERT_THRESHOLD,
            },
            'data': {'per_hour': alert_dict}}}}

    return top_level_alert_dict


def save_to_json(fp, d):

    with open(fp, 'w') as write_file:
        json.dump(d, write_file, indent=4)


if __name__ == '__main__':

    # assigned persons per building
    assignment_counts_dict = read_json('situational_awareness/data/synthetic_building_assignment_totals.json')

    # occupancy estimates
    # make sure the file below is at the correct location (you have to unzip it)
    occupancy_dict = read_json('local_outputs/situational_awareness/1619105129_results.json')

    # create alerts
    alert_dict = calculate_alert_dict(occupancy_dict, assignment_counts_dict['building_assignment_totals'])

    # save to json
    save_to_json('synthetic-alerts.json', alert_dict)
