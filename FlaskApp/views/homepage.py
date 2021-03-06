from flask import render_template
from FlaskApp import app


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['POST', 'GET'])
def home():
        return render_template("homepage.html")


@app.route('/about/')
def home_about():
    try:
        return render_template("about.html")
    except Exception as e:
        return(str(e))


@app.route('/contact/')
def home_contact():
    try:
        return render_template("contact.html")
    except Exception as e:
        return(str(e))


@app.route('/services/')
def services():
    try:
        return render_template("services.html")
    except Exception as e:
        return(str(e))