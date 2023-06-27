import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
from flask_restful import Resource, Api
from flask_api import status
import numpy as np
import pandas as pd


app = Flask("Late Payment Predictor")
api = Api(app=app)

model = pickle.load(open('../saved_model/classmodel.pkl', 'rb'))
scaler = pickle.load(open('../saved_model/scaling.pkl', 'rb'))

mappings = {'no': 0,
            'yes': 1,
            0: 'no',
            1: 'yes'}


class Classifier(Resource):

    def post(self):
        mandatory_params = ['international_plan',
                            'total_day_minutes',
                            'total_day_charge',
                            'total_eve_minutes',
                            'total_eve_charge',
                            'total_night_minutes',
                            'total_night_charge',
                            'total_intl_minutes',
                            'total_intl_charge',
                            'number_customer_service_calls']
        args = request.args

        for param in mandatory_params:
            if args.get(param) in [None, ""]:
                return f"Queryparam {param} nor defined", status.HTTP_400_BAD_REQUEST

        try:
            international_plan = args.get(
                'international_plan', default="no", type=str)
            total_day_minutes = args.get(
                'total_day_minutes', type=float)
            total_day_charge = args.get('total_day_charge', type=float)
            total_eve_minutes = args.get(
                'total_eve_minutes', type=float)
            total_eve_charge = args.get('total_eve_charge', type=float)
            total_night_minutes = args.get(
                'total_night_minutes', type=float)
            total_night_charge = args.get(
                'total_night_charge', type=float)
            total_intl_minutes = args.get(
                'total_intl_minutes', type=float)
            total_intl_charge = args.get(
                'total_intl_charge', type=float)
            number_customer_service_calls = args.get(
                'number_customer_service_calls', default=0, type=int)

            international_plan = mappings[international_plan.lower()]
            total_minutes = total_day_minutes + total_eve_minutes + \
                total_night_minutes + total_intl_minutes
            total_charge = total_day_charge + total_eve_charge + \
                total_night_charge + total_intl_charge

        except KeyError as err:
            return "international_plan should be yes/no", status.HTTP_400_BAD_REQUEST
        except TypeError as err:
            return "The total minutes and charge and the customer service calls should be numbers", status.HTTP_400_BAD_REQUEST

        classif_data_list = [international_plan,
                             total_day_minutes,
                             total_day_charge,
                             number_customer_service_calls,
                             total_minutes,
                             total_charge]
        classif_data_array = np.array(classif_data_list)

        # Scale the data and predict the classification
        scaled_data = scaler.transform(classif_data_array.reshape(1, -1))
        output = model.predict(scaled_data)
        output = mappings[int(output)]

        return jsonify(output)


api.add_resource(Classifier, '/api')

# @app.route('/api', methods=['POST'])
# def api():
#     data = request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
#     output=model.predict(new_data)
#     print(output)
#     return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
