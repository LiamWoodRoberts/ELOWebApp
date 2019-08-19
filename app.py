# Local Python Modules
import predictor
import utils
from elo_params import params

# Other Modules
from flask import Flask,render_template,request,jsonify,url_for,redirect,Blueprint
import requests
import json
from flask_restplus import Api,Resource,reqparse

app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
                
app.config.from_pyfile('./config/gunicorn.conf.py')

blueprint = Blueprint('api',__name__,url_prefix='/elo_api')
api_name = 'Customer Loyalty Prediction Model'
api = Api(blueprint,default=api_name,doc='/documentation')
app.register_blueprint(blueprint)

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
    
# APP ROUTES
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/demo")
def demo():
    sample_cols,sample_vals,metric_cols,metric_vals,preds,percentile = predictor.get_demo_values()
    return render_template("demo.html",
                            sample_cols=sample_cols,
                            sample_vals=sample_vals,
                            metric_cols=metric_cols,
                            metric_vals=metric_vals,
                            pred=preds,
                            percentile=percentile)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/nav")
def nav():
    return render_template("nav_bar.html")

# export FLASK_APP=app.py

if __name__ == "__main__":
    app.run()