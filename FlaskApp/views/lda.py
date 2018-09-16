from flask import render_template, request, url_for, redirect
from FlaskApp import app
import os
from FlaskApp.models.lda.lda_model import test
from FlaskApp.views.common_function import txt_normal, \
    get_name, four_digit, the_first_line, allowed_file

from werkzeug.utils import secure_filename

@app.route('/services/ldapred/', methods=['POST', 'GET'])
def lda_homepage():
    return render_template("ldapred/layout.html")


@app.route('/services/ldapred/about/')
def lda_about():
    try:
        return render_template("ldapred/about.html")
    except Exception as e:
        return(str(e))


@app.route('/services/ldapred/example/')
def lda_example():
    try:
        return render_template("ldapred/example.html")
    except Exception as e:
        return(str(e))


@app.route('/services/ldapred/data/')
def lda_data():
    try:
        return render_template("ldapred/data.html")
    except Exception as e:
        return(str(e))


@app.route('/services/ldapred/cancer-lectin/')
def lda_lda():
    try:
        return render_template("ldapred/cancer-lectin.html")
    except Exception as e:
        return(str(e))


@app.route('/services/ldapred/non-cancer-lectin/')
def lda_non_lda():
    try:
        return render_template("ldapred/non-cancer-lectin.html")
    except Exception as e:
        return(str(e))


@app.route('/services/ldapred/result/', methods=['POST', 'GET'])
def lda_result():
    if request.method == "POST":

        if request.form["lda"]:
            with open("/var/www/FlaskApp/input_data/ldapred.txt", 'w') as f:
                f.write(request.form['lda'])
            # 转换数据格式，变成一个新的文件
            test_data_path = "/var/www/FlaskApp/input_data/ldapred.txt"
        elif request.files['vir_file']:
            file = request.files['vir_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join("/var/www/FlaskApp/input_data/", filename))
                test_data_path = os.path.join("/var/www/FlaskApp/input_data/", filename)
        else:
            return render_template('ldapred/result.html')
        condition = False
        txt_normal(test_data_path)
        class_prob = []
        # 如果文件内有内容，继续
        first_line = the_first_line(test_data_path)
        if first_line == '\n':
            os.remove(test_data_path)
            return render_template('ldapred/result.html', class_prob=class_prob, condition=condition)
        else:
            condition = True
            seq_name = get_name(test_data_path)
            targets = test(test_data_path)
            targets = four_digit(targets)
            class_prob = enumerate(zip(seq_name, targets), start=1)
            os.remove(test_data_path)
            return render_template('ldapred/result.html', class_prob=class_prob, condition=condition)

    return render_template('ldapred/result.html')
