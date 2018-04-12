from flask import Flask
from flask_bootstrap import Bootstrap

app = Flask(__name__)

app.config['SECRET_KEY'] = "somestring"
bootstrap = Bootstrap(app)

from FlaskApp.views import antioxidant
# from FlaskApp.views import cancerlectin
from FlaskApp.views import bacteriophage
from FlaskApp.views import homepage

