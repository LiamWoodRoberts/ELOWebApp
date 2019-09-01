# Module Imports
from app import api,predictor,utils
from app.elo_params import params

# Package Imports
from flask_restplus import Resource,reqparse
from flask import render_template,request,jsonify,url_for,redirect
import requests
import json

# API ROUTES
@api.route("/sample")
class random_sample(Resource):
    def get(self):
        '''returns a random sample from the original elo dataset'''
        sample = predictor.get_sample(1)
        return sample.to_json()

# Arguments for metrics,eval,and predict api post requests
parser = reqparse.RequestParser()
parser.add_argument('sample',type=str)

@api.route("/metrics")
class metrics(Resource):
    def get(self):
        '''returns key model metrics from a random sample'''
        sample_df = predictor.get_sample(1)
        customer_metrics = predictor.get_metrics(sample_df)
        return customer_metrics

    def post(self):
        '''returns key model metrics from a defined sample'''
        args = parser.parse_args()
        sample = str(args['sample'])
        sample_df = utils.parse_post(sample)
        customer_metrics = predictor.get_metrics(sample_df)
        return customer_metrics

@api.route("/eval")
class evaluate(Resource):
    def get(self):
        '''returns model predictions and actual values from a random sample'''
        sample_df = predictor.get_sample(1)
        features = predictor.get_features(sample_df)
        preds,target = predictor.get_sample_eval(sample_df)
        return jsonify(preds=preds.tolist(),target=target.tolist())

    def post(self):
        '''returns model predictions and actual values from a defined sample'''
        args = parser.parse_args()
        sample = str(args['sample'])
        sample_df = utils.parse_post(sample)
        preds,target = predictor.get_sample_eval(sample_df)
        return jsonify(preds=preds.tolist(),target=target.tolist())

@api.route("/predict")
class predict(Resource):
    def get(self):
        '''returns model predictions for a random sample'''
        sample_df = predictor.get_sample(1)
        preds = predictor.get_prediction(sample_df)
        return jsonify(preds.tolist())

    def post(self):
        '''returns model predictions for a defined sample'''
        args = parser.parse_args()
        sample = str(args['sample'])
        sample_df = utils.parse_post(sample)
        preds = predictor.get_prediction(sample_df)
        return jsonify(round(preds.tolist()[0],2))

# arguments for percentile api post requests
pct_parser = reqparse.RequestParser()
pct_parser.add_argument('pred')

@api.route("/percentiles")
class percentiles(Resource):
    def post(self):
        '''returns percentiles for predicted customer loyatly'''
        args = pct_parser.parse_args()
        pred = float(args['pred'])
        percentile = predictor.get_percentile(pred)
        return percentile 