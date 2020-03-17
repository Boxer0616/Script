import csv
import pprint
import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outfilepath', type=str , default='./train.csv')
parser.add_argument('-t', '--outfile', type=str , default='./train.csv')
parser.add_argument('-f', '--inputfilepath', type=str , default='./800polyclop/train/226200_104900')
args = parser.parse_args()

#csvファイル作成
def fileopen(file_name):
        train,test = list_split(file_name)
        file_list = []
        for file in file_name:
                file_list.append(file)
                #with open(args.outfilepath, 'w') as f:
                 #       writer = csv.writer(f)
                        #writer.writerow([0, 1, 2])
                        #writer.writerow(['a', 'b', 'c'])
                  #      writer.writerow(file)
        test = file_list[0:len(file_list)]
        with open(args.outfilepath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(test)
        #with open(args.outfilepath) as f:
         #       print(f.read())

def list_split(file_name):
        #ファイルを取得してlistにいれる
        file_list = []
        for file in file_name:
                file_list.append(file)
        
        #リストをシャッフル
        random.shuffle(file_list)
        #分割の割合
        ratio = 0.7
        #分割
        p = int(ratio * len(file_list)) # 分割点を計算
        train = file_list[0:p]
        test = file_list[p:]
                
        return train,test

#ファイルの名前取得
def get_name():
    path = args.inputfilepath
    files = os.listdir(path)
    files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
    # ファイルの並びをソート
    s_flist = sorted(files_file)
    #print(type(files))
    #print(files_file)
    return s_flist

if __name__ == '__main__':
    file_name = get_name()
    fileopen(file_name)
