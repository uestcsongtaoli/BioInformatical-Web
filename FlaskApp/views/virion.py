from flask import render_template, request, url_for, redirect
from FlaskApp import app
import os
from FlaskApp.models.virion import model
from FlaskApp.views.common_function import txt_normal, \
    get_seq_name, four_digit, the_first_line, allowed_file
from FlaskApp.views.forms import InputForm, FileForm
from werkzeug.utils import secure_filename

@app.route('/virionpred/', methods=['POST', 'GET'])
def virion_homepage():
    return render_template("virionpred/layout.html")


@app.route('/virionpred/about/')
def virion_about():
    try:
        return render_template("virionpred/about.html")
    except Exception as e:
        return(str(e))


@app.route('/virionpred/example/')
def virion_example():
    try:
        return render_template("virionpred/example.html")
    except Exception as e:
        return(str(e))


@app.route('/virionpred/data/')
def virion_data():
    try:
        return render_template("virionpred/data.html")
    except Exception as e:
        return(str(e))


@app.route('/virionpred/virion/')
def virion_virion():
    try:
        return render_template("virionpred/virion.html")
    except Exception as e:
        return(str(e))


@app.route('/virionpred/non-virion/')
def virion_non_virion():
    try:
        return render_template("virionpred/non-virion.html")
    except Exception as e:
        return(str(e))


@app.route('/virionpred/result/', methods=['POST', 'GET'])
def virion_result():
    if request.method == "POST":

        if request.form["virion"]:
            with open("/var/www/FlaskApp/input_data/virion.txt", 'w') as f:
                f.write(request.form['virion'])
            # 转换数据格式，变成一个新的文件
            test_data_path = "/var/www/FlaskApp/input_data/virion.txt"
        elif request.files['vir_file']:
            file = request.files['vir_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join("/var/www/FlaskApp/input_data/", filename))
                test_data_path = os.path.join("/var/www/FlaskApp/input_data/", filename)
        else:
            return render_template('virionpred/result.html')
        condition = False
        txt_normal(test_data_path)
        class_prob = []
        # 如果文件内有内容，继续
        first_line = the_first_line(test_data_path)
        if first_line == '\n':
            os.remove(test_data_path)
            return render_template('virionpred/result.html', class_prob=class_prob, condition=condition)
        else:
            condition = True
            seq_name = get_seq_name(test_data_path)
            targets = model.predict(test_data_path)
            targets = four_digit(targets)
            class_prob = enumerate(zip(seq_name, targets), start=1)
            os.remove(test_data_path)
            return render_template('virionpred/result.html', class_prob=class_prob, condition=condition)

    return render_template('virionpred/result.html')
