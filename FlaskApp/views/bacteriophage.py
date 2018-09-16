from flask import render_template, request
from FlaskApp.models.bacteriophage import multinomialNB
from FlaskApp import app
from FlaskApp.views.common_function import txt_normal, \
    get_seq_name, four_digit, the_first_line


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
        if request.form["bacteriophage"]:
            condition = False
            with open("/var/www/FlaskApp/input_data/bacteriophage.txt", 'w') as f:
                f.write(request.form['bacteriophage'])
            # 转换数据格式，变成一个新的文件
            test_data_path = "/var/www/FlaskApp/input_data/bacteriophage.txt"
            txt_normal(test_data_path)
            class_prob =[]
            # 如果文件内有内容，继续
            first_line = the_first_line(test_data_path)
            if first_line == '\n':
                return render_template('bacteriophage/result.html', class_prob=class_prob, condition=condition)
            else:
                condition = True
                seq_name = get_seq_name(test_data_path)
                targets = multinomialNB.train_test(train_data_path, test_data_path, test_save_data_path)

                targets = four_digit(targets)
                class_prob = enumerate(zip(seq_name, targets), start=1)

                return render_template('bacteriophage/result.html', class_prob=class_prob, condition=condition)

    return render_template('bacteriophage/result.html')







