from flask import render_template, request
from FlaskApp import app
from FlaskApp.models.antioxidant import t_sne


@app.route('/IDAod/', methods=['POST', 'GET'])
def antioxidant_homepage():
    try:
        return render_template("antioxidant/layout.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/about/')
def antioxidant_about():
    try:
        return render_template("antioxidant/about.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/example/')
def antioxidant_example():
    try:
        return render_template("antioxidant/example.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/data/')
def antioxidant_data():
    try:
        return render_template("antioxidant/data.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/antioxidant/')
def antioxidant_antioxidant():
    try:
        return render_template("antioxidant/antioxidant.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/non-antioxidant/')
def antioxidant_non_antioxidant():
    try:
        return render_template("antioxidant/non-antioxidant.html")
    except Exception as e:
        return(str(e))


@app.route('/IDAod/result/', methods=['POST', 'GET'])
def antioxidant_result():
    if request.method == "POST":
        with open(r'/var/www/FlaskApp/input_data/antioxidant.txt', 'w') as f:
            f.write(request.form['antioxidant'])

        if request.form["antioxidant"][0] == '>':
            targets = t_sne.result()
        else:
            targets = "You should have a '>' before every sequence."

        return render_template('antioxidant/result.html', sequence=targets)
    return render_template('antioxidant/result.html')

