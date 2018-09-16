
import re
import pandas as pd

def get_seq_name(path):
    seq_name = []
    for line in open(path):
        line = line.strip('\n')
        if line != '':
            if line[0] == ">":
                line = line[1:]
                seq_name.append(line)
    return seq_name



def get_name(path):
    seq_name = []
    for line in open(path):
        line = line.strip('\n')
        if line != '':
            if line[0] == ">":
                line = line[1:]
                for num, i in enumerate(line):
                    if i.isspace():
                        line = line[:num]
                        break
                seq_name.append(line)
    return seq_name



def four_digit(result):
    four_bit_result = []
    item = []
    for i, j in result:
        item.append(format(i, '.4f'))
        item.append(format(j, '.4f'))
        four_bit_result.append(item)
        item =[]
    return four_bit_result


def txt_normal(file):
    f = open(file, "r")
    lines = f.readlines()
    a = []
    for line in lines:
        if line.split():
            if line != "\n" and line != "\r":
                line_non_n = line.strip()
                if line_non_n[0] != ">":
                    line_nor = re.sub("[^A-Z]", "", line_non_n)
                    if line_nor != '':
                        a.append(line_nor.strip())
                else:
                    a.append(line_non_n)
    string = ""
    b = []
    for i in a:
        if i != '':
            if i[0] == ">":
                if string != "":
                    b.append(string)
                    string = ""
                b.append(i)
            else:
                string = string + i
    b.append(string)
    fp = open(file, "w+")
    for i in b:
        fp.write(i+"\n")
    fp.close()


def the_first_line(path):
    fp = open(path)
    for num, line in enumerate(fp):
        if num == 0:
            first_line = line
    fp.close()
    return first_line


ALLOWED_EXTENSIONS = set(['txt'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pos_samples(input_file, new_file, pos_list):
    import math

    file = open(input_file)
    i = 1
    pos = []
    for line in file.readlines():
        line = line.strip('\n')
        if math.ceil(i / 2) in pos_list:
            pos.append(line)
        i += 1
    file.close()

    fp = open(new_file, "w+")
    for i in pos:
        fp.write(i + "\n")
    fp.close()


def merge_two(result1, result2, pos_nums):
    s1 = pd.Series(result1, index=range(1, len(result1)+1))
    s2 = pd.Series(result2, index=pos_nums)
    s1[s2.index] = s2.values

    return list(s1)


def pos_neg(file):
    import math
    f = open(file)
    i = 1
    neg_list = []
    for line in f.readlines():
        line = line.strip('\n')
        if line[1] == 'a':
            neg_list.append(math.ceil(i/2))
        i += 1
    f.close()
    return neg_list


def merge_result(result1, result2, neg_nums):
    s1 = pd.Series(result1, index=range(1, len(result1)+1))
    s2 = pd.Series(result2, index=range(1, len(result2)+1))
    s1[neg_nums] = s2[neg_nums].values

    return list(s1)


def change_result(pos_samples, result):
    result = pd.Series(result, index=range(1, len(result)+1))
    for k, i in enumerate(result[pos_samples]):
        if i[0] > i[1]:
            result[pos_samples[k]] = (i[1], i[0])
    return list(result)