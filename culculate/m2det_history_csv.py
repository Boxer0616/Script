# -*- coding: utf-8 -*-

# pylint: disable = W0621, W0622
# pylint: disable = C0111, C0413, C0103, C0326, C0325, C0304, C0411

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--csv_path', default='../output/history/loss.csv')
args = parser.parse_args()


"""
-----------------------------------------------------------------------------------------

結果を読み込み

-----------------------------------------------------------------------------------------
"""
def load_results(filename):

    result_list = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # ヘッダーを飛ばす
        header = next(reader)

        for row in reader:
            itr    = int(row[3])
            loss_l = float(row[4])
            loss_c = float(row[5])
            loss   = float(row[4]) + float(row[5])

            result_list.append([itr, loss_l, loss_c, loss])

    return result_list


"""
-----------------------------------------------------------------------------------------

プロット

-----------------------------------------------------------------------------------------
"""
def plot_loss_and_accuracy(result_list, save_dir):

    itr_list    = [row[0] for row in result_list]
    loss_l_list = [row[1] for row in result_list]
    loss_c_list = [row[2] for row in result_list]
    loss_list   = [row[3] for row in result_list]

    # 単純移動平均線
    loc_y_aves  = np.convolve(loss_l_list, np.ones(50)/float(50), 'valid')
    loc_x_aves  = np.arange(1, itr_list[-1], (itr_list[-1] - 1)/len(loc_y_aves))

    conf_y_aves = np.convolve(loss_c_list, np.ones(50)/float(50), 'valid')
    conf_x_aves = np.arange(1, itr_list[-1], (itr_list[-1] - 1)/len(conf_y_aves))

    loss_y_aves = np.convolve(loss_list, np.ones(50)/float(50), 'valid')
    loss_x_aves = np.arange(1, itr_list[-1], (itr_list[-1] - 1)/len(loss_y_aves))

    plt.figure()
    #plt.plot(itr_list, loss_c_list, 'g-', label = 'loss_C')
    #plt.plot(itr_list, loss_l_list, 'r-', label = 'loss_L')
    plt.plot(itr_list, loss_list, label = 'loss')
    #aa = np.delete(loss_x_aves, 1244950)
    plt.plot(loss_x_aves, loss_y_aves, 'r-', label = 'loss(avg)')
    #plt.plot(aa, loss_y_aves, 'r-', label = 'loss(avg)')
    plt.grid()
    plt.legend()
    plt.xlabel('itelation')
    plt.ylabel('loss')
    plt.ylim((0, 100))
    plt.savefig(os.path.join(save_dir, 'loss.png'))


"""
-----------------------------------------------------------------------------------------

メイン関数

-----------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    result_list = load_results(args.csv_path)

    save_dir = args.csv_path.rsplit('/', 1)[0]
    plot_loss_and_accuracy(result_list, save_dir)

    print ("Successful Completion")
