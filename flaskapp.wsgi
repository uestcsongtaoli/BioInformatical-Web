import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, "/var/www/FlaskApp/")

from FlaskApp import app  as application
# # The secret key
application.secret_key = 'tpctbptp02'
