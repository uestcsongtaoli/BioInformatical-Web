from flask import render_template, request
from FlaskApp import app
import os
from FlaskApp.models.virion import model

absolute_path = os.path.abspath('.')

@app.route('/virion/', methods=['POST', 'GET'])
def virion_homepage():
    try:
        return render_template("virion/layout.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/about/')
def virion_about():
    try:
        return render_template("virion/about.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/example/')
def virion_example():
    try:
        return render_template("virion/example.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/data/')
def virion_data():
    try:
        return render_template("virion/data.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/virion/')
def virion_virion():
    try:
        return render_template("virion/virion.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/non-virion/')
def virion_non_virion():
    try:
        return render_template("virion/non-virion.html")
    except Exception as e:
        return(str(e))


@app.route('/virion/result/', methods=['POST', 'GET'])
def virion_result():
    if request.method == "POST":
        with open(absolute_path + '/input_data/virion.txt', 'w') as f:
            f.write(request.form['virion'])
        test_data_path = absolute_path + '/input_data/virion.txt'
        if request.form["virion"][0] == '>':
            targets = model.predict(test_data_path)
        else:
            targets = "You should have a '>' before every sequence."

        return render_template('virion/result.html', sequence=targets)
    return render_template('virion/result.html')

