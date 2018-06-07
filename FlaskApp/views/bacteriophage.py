from flask import render_template, request
from FlaskApp.models.bacteriophage import multinomialNB
from FlaskApp import app
import os
import sys
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
from FlaskApp.views.forms import FileForm


# absolute_path = os.path.abspath('.')
#   整个项目的根文件路径'/var/www/FlaskApp'

@app.route('/bacteriophage/', methods=['POST', 'GET'])
def bact_homepage():
    # form = FileForm()
    try:
        return render_template("bacteriophage/layout.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/about/')
def bact_about():
    try:
        return render_template("bacteriophage/about.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/example/')
def bact_example():
    try:
        return render_template("bacteriophage/example.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/data/')
def bact_data():
    try:
        return render_template("bacteriophage/data.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/bact/')
def bact_bact():
    try:
        return render_template("bacteriophage/bact.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/non-bact/')
def bact_non_bact():
    try:
        return render_template("bacteriophage/non-bact.html")
    except Exception as e:
        return(str(e))


@app.route('/bacteriophage/result/', methods=['POST', 'GET'])
def bact_result():
    if request.method == "POST":
        train_data_path = "/var/www/FlaskApp/FlaskApp/models/bacteriophage/data/"
        test_save_data_path = "/var/www/FlaskApp/FlaskApp/models/bacteriophage/test_data_save/"
        # print(request.files['file'])
        # print(not request.files['file'])
        # print(request.form["bacteriophage"])

        if request.form["bacteriophage"]:
            with open("/var/www/FlaskApp/input_data/bacteriophage.txt", 'w') as f:
                f.write(request.form['bacteriophage'].replace('', ''))

            if request.form["bacteriophage"][0] == '>':
                test_data_path = "/var/www/FlaskApp/input_data/bacteriophage.txt"
                targets = multinomialNB.train_test(train_data_path, test_data_path, test_save_data_path)
            else:
                targets = "The sequences you input should be in FASTA format."\
                          "And there should not be any space line on the top of textarea"

        # if request.files['file']:
        #     file = request.files['file']
        #     file.save(absolute_path + '/input_data/test.txt')
        #     test_data_path = absolute_path + '/input_data/test.txt'
        #     targets = multinomialNB.train_test(train_data_path, test_data_path, test_save_data_path)

        # if request.files['file'] is None and request.form["bacteriophage"] is None:
        #     targets = "You may need to enter query sequences in the textarea or upload a file for batch prediction. "

        return render_template('bacteriophage/result.html', sequence=targets)
    return render_template('bacteriophage/result.html')

