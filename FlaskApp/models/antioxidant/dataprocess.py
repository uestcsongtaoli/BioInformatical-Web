# -*- coding:utf-8 -*-
import itertools
import multiprocessing
from collections import Counter
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import Pool

import numpy
import pandas
import re

w=('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y')

def load_data(path):
    '''
    :param path:the destination of the protein sequence

    :return:the protein sequence
    '''

    list = []

    with open(path) as file:
        for line in file.readlines():
            if line.find('>') == -1:
                list.append(line.strip('\n'))

    return list


def Dictionary(w, dipe=1):

    dictionary = []

    iter = itertools.product(w, repeat=dipe)
    for elem in iter:
        dictionary.append(''.join(elem))

    return dictionary


def Laplace_smootching(motif, samples_index):

    counter = Counter(motif)
    unique = list(set(motif))

    dictionary = Dictionary(w, len(motif[0]))
    frequent = numpy.zeros(len(dictionary))
    all = sum(counter.values())

    for item in unique:

        count = counter.get(item)
        frequent[dictionary.index(item)] = (count) / all


    return frequent
    # dataframe = pandas.DataFrame(frequent)
    # path = 'dipe'+str(len(motif_all_samples[0])+'_gap'+str(gap)+'.csv') #保存的路径名
    # dataframe.to_csv(path, header=False, index=False)


def frequence(save_data, motif_all_samples, gap, mode=-1):

    frequent = numpy.zeros((len(motif_all_samples), 20**(len(motif_all_samples[0][0]))))

    for i in range(len(motif_all_samples)):

        frequent[i] = Laplace_smootching(motif_all_samples[i], i)

    # dataframe = pandas.DataFrame(frequent)
    # if mode != -1:
    #     # path = 'dipe' + str(len(motif_all_samples[0][0])) +'_gap' + str(gap)+ '_'+str(mode) + '.csv' #保存的路径名
    #     path = save_data + 'gap'+ str(gap)+'_dipe'+str(len(motif_all_samples[0][0]))+str(mode)+'.csv'
    # else:
    #     path = save_data + 'gap' + str (gap) + '_dipe' + str (len (motif_all_samples[0][0])) + '.csv'
    # print('save_path:', path)
    # dataframe.to_csv(path, header=None, index=False)
    return frequent


def get_gap_motif(motif_all_samples, mode):

    motif_new = []
    for motif in motif_all_samples:

        if mode == 0:
            i = [item[0] + item[-1] for item in motif]
        elif mode == 1:
            i = [item[0] + item[2:] for item in motif]
        elif mode == 2:
            i = [item[0:2] + item[-1] for item in motif]
        elif mode == 3:
            i = [item[::2] for item in motif]
        elif mode == 4:
            i = [item[0] + item[-2:] for item in motif]
        elif mode == 5:
            i = [item[0:2] + item[-1] for item in motif]

        motif_new.append(i)

    return motif_new


def map_func(sequence, pattern):

    pattern = re.compile(pattern)
    motif_all_samples = list(pattern.findall(sequence))

    return motif_all_samples

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def main(protein, save_data, pattern_list=['(?=(.))', '(?=(..))', '(?=(...))', '(?=(....))', '(?=(.....))']):

    dict = {}
    for i in range(len(pattern_list)):

        # pattern = re.compile(pattern_list[i])

        # motif_all_samples = list(pattern.findall(sequence) for sequence in protein)
        with poolcontext(processes=3) as pool:
            motif_all_samples = pool.map(partial(map_func, pattern=pattern_list[i]), protein)

        if len(motif_all_samples[0][0]) == 3:

            dict['gap'+ str(0)+'_dipe'+str(len(motif_all_samples[0][0]))] = frequence(save_data, motif_all_samples, gap=0)

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=0)  # A*A
            dict['gap' + str(1) + '_dipe' + str (len (motif_all_samples_1[0][0]))] = frequence(save_data, motif_all_samples_1, gap=1)

        elif len(motif_all_samples[0][0]) == 4:

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=1)  # A*AA
            dict['gap' + str (1) + '_dipe' + str(len(motif_all_samples_1[0][0]))+str(1)] = frequence(save_data, motif_all_samples_1, gap=1, mode=1)

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=2) #AA*A
            dict['gap' + str (1) + '_dipe' + str (len (motif_all_samples_1[0][0])) + str (2)] = frequence(save_data, motif_all_samples_1, gap=1, mode=2)

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=0) # A**A
            dict['gap' + str (2) + '_dipe' + str (len (motif_all_samples_1[0][0]))] = frequence(save_data, motif_all_samples_1, gap=2)

        elif len(motif_all_samples[0][0]) == 5:

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=3)  # A*A*A
            dict['gap' + str (2) + '_dipe' + str (len (motif_all_samples_1[0][0])) + str (1)] = frequence (save_data, motif_all_samples_1, gap=2, mode=1)

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=4)  # A**AA
            dict['gap' + str (2) + '_dipe' + str(len (motif_all_samples_1[0][0]))+str (2)] = frequence (save_data, motif_all_samples_1, gap=2, mode=2)

            motif_all_samples_1 = get_gap_motif(motif_all_samples, mode=5)  # AA**A
            dict['gap' + str (2) + '_dipe' + str (len (motif_all_samples_1[0][0])) + str (3)] =frequence (save_data, motif_all_samples_1, gap=2, mode=3)

        else:

            dict['gap' + str(0) + '_dipe' + str(len (motif_all_samples[0][0]))] =frequence(save_data, motif_all_samples, gap=0)

    dataframe = dict['gap0_dipe1']
    dataframe = numpy.concatenate([dataframe, dict.get('gap0_dipe2'), dict.get('gap0_dipe3'), dict.get('gap1_dipe2'),
                                                    dict.get('gap2_dipe31'), dict.get('gap2_dipe2'), dict.get('gap2_dipe32'),
                                                    dict.get('gap1_dipe32'), dict.get('gap1_dipe31'), dict.get('gap2_dipe33')], axis=1)
    # dataframe.to_csv(save_data+'test_data.csv', header=None, index=False)
    return dataframe


if __name__=='__main__':

    protein = load_data(r'E:\python\data\phage\data\virion1.txt')
    protein.extend(load_data(r'E:\python\data\phage\data\non-virion1.txt'))
    pattern_list = ['(?=(.))', '(?=(..))', '(?=(...))', '(?=(....))', '(?=(.....))']
    main(protein, pattern_list)
