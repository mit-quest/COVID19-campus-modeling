from flask import render_template, redirect
import json
import os

# Main UI endpoints
# direct route for '/' from main.py


def main():
    return render_template('main.html')


def forecast():
    return render_template('forecast.html')


def situational_awareness():
    return render_template('situational-awareness.html')

# Secondary endpoints used by UI endpoints to pull data from server
# direct route for '/get-uuids/forecast' from main.py - called from within JavaScript of '/forecast'


def get_forecast_uuids(request):

    try:
        files = os.listdir('../local_outputs/scenarios')
    except Exception:
        return 'failed to read files'

    values = [os.path.splitext(file)[0].replace('_metadata', '') for file in files]

    return '<br/>'.join(values)

# direct route for '/get-data/forecast?uuid=<uuid>' from main.py - called from within JavaScript of '/forecast'


def get_forecast_data(request):
    print('start get_data')

    submitted = request.form

    if 'uuid' in submitted:
        uuid = submitted['uuid']
    else:
        return {'failure-message': 'failed - no uuid'}

    scriptpath = os.path.dirname(__file__)

    outputDir = os.path.join(os.path.dirname(os.path.dirname(scriptpath)), 'local_outputs')
    if not os.path.isdir(outputDir):
        return {'failure-message': 'failed - no local_outputs directory'}

    combinedResults = {}
    combinedResults['scenario'] = {}
    combinedResults['analysis'] = {}

    path = os.path.join(outputDir, 'scenarios', uuid + '_metadata.json')
    with open(path, 'r') as file:
        combinedResults['scenario']['metadata'] = json.load(file)

    types = os.listdir(os.path.join(os.path.dirname(os.path.dirname(scriptpath)), 'local_outputs', 'analyses'))

    for type in types:

        pathToType = os.path.join(outputDir, 'analyses', type, 'V0.1')
        if os.path.isdir(pathToType):
            combinedResults['analysis'][type] = {}

            path = os.path.join(pathToType, uuid + '-' + type + '_metadata.json')
            with open(path, 'r') as file:
                combinedResults['analysis'][type]['metadata'] = json.load(file)

            path = os.path.join(pathToType, uuid + '-' + type + '_results.json')
            with open(path, 'r') as file:
                combinedResults['analysis'][type]['results'] = json.load(file)

    return json.dumps(combinedResults)


def get_situational_awareness_uuids(request):

    try:
        files = os.listdir('../local_outputs/situational_awareness')
    except Exception:
        return 'failed to read files'

    validUUIDs = []
    for file in files:
        if '_metadata' in file:
            validUUIDs.append(file.replace('_metadata.json', ''))

    return '<br/>'.join(validUUIDs)


def get_situational_awareness_data(request):
    print('start get_data')

    submitted = request.form

    if 'uuid' in submitted:
        uuid = submitted['uuid']
    else:
        return {'failure-message': 'failed - no uuid'}

    scriptpath = os.path.dirname(__file__)

    outputDir = os.path.join(os.path.dirname(os.path.dirname(scriptpath)), 'local_outputs')
    if not os.path.isdir(outputDir):
        return {'failure-message': 'failed - no local_outputs directory'}

    combinedResults = {}

    path = os.path.join(outputDir, 'situational_awareness', uuid + '_metadata.json')
    with open(path, 'r') as file:
        combinedResults['metadata'] = json.load(file)

    path = os.path.join(outputDir, 'situational_awareness', uuid + '_results.json')
    with open(path, 'r') as file:
        combinedResults['results'] = json.load(file)

    return json.dumps(combinedResults)
