
import datetime
import json
import platform

from flask import Flask, render_template, request, make_response
from flask_compress import Compress

import handlers.main_view

app = Flask(__name__)
Compress(app)

# Main UI endpoints
# displays a list of results/<uuid>.json files that can be turned into reports


@app.route('/', methods=['GET'])
def view1():
    output = handlers.main_view.main()
    return output


@app.route('/forecast', methods=['GET'])
def view2():
    output = handlers.main_view.forecast()
    return output


@app.route('/situational-awareness', methods=['GET'])
def view3():
    output = handlers.main_view.situational_awareness()
    return output

# simple test page to check server


@app.route('/ping', methods=['GET'])
def ping():
    return 'hello world'

# Secondary endpoints used by UI endpoints to pull data from server
# used by static/js/forecast/app-ops.js


@app.route('/get-uuids/forecast', methods=['GET'])
def view4():
    output = handlers.main_view.get_forecast_uuids(request)
    return output

# used by static/js/forecast/app-ops.js


@app.route('/get-data/forecast', methods=['POST'])
def view5():
    output = handlers.main_view.get_forecast_data(request)
    return output

# used by static/js/situational_awarenes/app-ops.js


@app.route('/get-uuids/situational-awareness', methods=['GET'])
def view6():
    output = handlers.main_view.get_situational_awareness_uuids(request)
    return output

# used by static/js/situational_awarenes/app-ops.js


@app.route('/get-data/situational-awareness', methods=['POST'])
def view7():
    output = handlers.main_view.get_situational_awareness_data(request)
    return output

# run app


app.run(host='127.0.0.1', port='8080', debug=True)
