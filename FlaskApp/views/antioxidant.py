from flask import render_template, request
from FlaskApp import app
from FlaskApp.models.antioxidant import t_sne
from FlaskApp.models.antioxidant.antioxidant_server import predict
from FlaskApp.models.antioxidant.multinomialNB import train_test
import os
from FlaskApp.views.common_function import txt_normal, get_seq_name
from FlaskApp.views.common_function import four_digit, the_first_line
from FlaskApp.views.common_function import allowed_file, pos_samples
from FlaskApp.views.common_function import merge_two, pos_neg
from FlaskApp.views.common_function import merge_result, change_result
from werkzeug.utils import secure_filename


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
        train_data_path = "/var/www/FlaskApp/FlaskApp/models/antioxidant/data/"
        test_save_data_path = "/var/www/FlaskApp/FlaskApp/models/antioxidant/test_data_save/"
        if request.form["antioxidant"]:
            with open("/var/www/FlaskApp/input_data/antioxidant.txt", 'w') as f:
                f.write(request.form['antioxidant'])
            # 转换数据格式，变成一个新的文件
            test_data_path = "/var/www/FlaskApp/input_data/antioxidant.txt"
        elif request.files['ant_file']:
            file = request.files['ant_file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join("/var/www/FlaskApp/input_data/", filename))
                test_data_path = os.path.join("/var/www/FlaskApp/input_data/", filename)
        else:
            return render_template('antioxidant/result.html')
        condition = False
        txt_normal(test_data_path)
        class_prob = []
        # 如果文件内有内容，继续
        first_line = the_first_line(test_data_path)
        if first_line == '\n':
            os.remove(test_data_path)
            return render_template('antioxidant/result.html', class_prob=class_prob, condition=condition)
        else:
            condition = True
            seq_name = get_seq_name(test_data_path)
            # targets1, targets2 = t_sne.result(test_data_path), train_test(train_data_path, test_data_path, test_save_data_path)
            # neg_list = pos_neg(test_data_path)
            # targets = merge_result(targets1, targets2, neg_list)

            # print(targets1)
            # pos_nums = []
            # for num, i in enumerate(targets1, start=1):
            #     if i[1] >= 0.5:
            #         pos_nums.append(num)
            # print(pos_nums)
            # if len(pos_nums) > 0:
            #     new_file = "/var/www/FlaskApp/input_data/antioxidant_new.txt"
            #     pos_samples(test_data_path, new_file, pos_nums)
            #     targets2 = train_test(train_data_path, new_file, test_save_data_path)
            #     print(targets2)
            #     targets = merge_two(targets1, targets2, pos_nums)
            #
            # else:
            #     targets = targets1
            targets1 = train_test(train_data_path, test_data_path, test_save_data_path)
            pos_list = pos_neg(test_data_path)
            if len(pos_list) > 0:
                targets2 = change_result(pos_list, targets1)
            else:
                targets2 = targets1

            targets = four_digit(targets2)
            class_prob = enumerate(zip(seq_name, targets), start=1)
            os.remove(test_data_path)
            return render_template('antioxidant/result.html', class_prob=class_prob, condition=condition)

    return render_template('antioxidant/result.html')
