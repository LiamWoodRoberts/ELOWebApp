# Other Modules
from flask import Flask,Blueprint
from flask_restplus import Api

app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

app.config.from_object('config.Config')

blueprint = Blueprint('api',__name__,url_prefix='/elo_api')
api_name = 'Customer Loyalty Prediction Model'
api = Api(blueprint,default=api_name,doc='/documentation')
app.register_blueprint(blueprint)
    
from app import views
from app import api_views
