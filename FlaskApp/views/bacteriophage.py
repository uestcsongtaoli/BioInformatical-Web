from flask import render_template, request
from FlaskApp.models.bacteriophage import multinomialNB
from FlaskApp import app


@app.route('/bacteriophage/', methods=['POST', 'GET'])
def bact_homepage():
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
        with open(r'/var/www/FlaskApp/input_data/bacteriophage.txt', 'w') as f:
            f.write(request.form['bacteriophage'].replace('', ''))

        if request.form["bacteriophage"][0] == '>':
            train_data_path = "/var/www/FlaskApp/FlaskApp/models/bacteriophage/data/"
            test_data_path = '/var/www/FlaskApp/input_data/bacteriophage.txt'
            test_save_data_path = '/var/www/FlaskApp/FlaskApp/models/bacteriophage/test_data_save/'
            targets = multinomialNB.train_test(train_data_path, test_data_path, test_save_data_path)
        else:
            targets = "You should have a '>' before every sequence."

        return render_template('bacteriophage/result.html', sequence=targets)
    return render_template('bacteriophage/result.html')

