# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
import sys
import glob
from PIL import Image
import openpyxl as px
from openpyxl.styles import Alignment, borders
from openpyxl.styles.fonts import Font
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser(description = 'Caluclate IOU and Detection rate')

parser.add_argument('-c', '--category', help = 'Path to category text file', type = str, default = './category.txt')
parser.add_argument('-px', '--pre_xml', help = 'Path to predict xml directory', type = str, default = './output-nms/640/')
parser.add_argument('-cx', '--cor_xml', help = 'Path to correct xml directory', type = str, default = '/home/hiroshi/Dataset/folda-change/20191011矩形最終/val')
parser.add_argument('-na', '--name',type = str, default = '512prselfnms')
parser.add_argument('-t', '--thres',type = float, default = 1.1)
parser.add_argument('-i', '--iou_threshold', help = 'Threshold of IoU', type = float, default = 0.1)

args = parser.parse_args()


"""
---------------------------------------------------------------------------------------------------

XMLファイルからラベルを取得

---------------------------------------------------------------------------------------------------
"""
def get_label(xml_list, correct = False):
    label_list = []

    for xml_path in xml_list:
        tree = ET.parse(xml_path)
        directory = xml_path.rsplit('/', 3)[1]
        framename = xml_path.rsplit('/', 3)[3].rsplit('.', 1)[0]
        dir_frame = os.path.join(directory, framename)

        num = 0

        for object in tree.findall('object'):
            labelname = object.find('category').find('value').text

            if not (correct):
                score = object.find('score').text

            for bndbox in object.findall('bndbox'):
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)

                if not (correct):
                    label_list.append([dir_frame, labelname, xmin, ymin, xmax, ymax, num, score])

                else:
                    label_list.append([dir_frame, labelname, xmin, ymin, xmax, ymax, num])

                # １枚の画像に含まれるバウンディングボックス数をカウント
                num += 1

    # label_list = [directory/framename, labelname, x1, y1, x2, y2, num, (score)]
    return label_list


"""
---------------------------------------------------------------------------------------------------

ボックス合計数を取得

---------------------------------------------------------------------------------------------------
"""
def get_boxnum(label_list, categories):
    boxnum_list = []

    for category in categories:
        cnt = [x[1] for x in label_list].count(category)

        boxnum_list.append([category, cnt])

    return boxnum_list


"""
---------------------------------------------------------------------------------------------------

IOUを計算

---------------------------------------------------------------------------------------------------
"""
def calculate_iou(predict_label_list, correct_label_list):
    iou_list = []

    for cor in correct_label_list:
        for pre in predict_label_list:
            if (cor[0] == pre[0]):
                K_area = (int(pre[4]) - int(pre[2]) + 1) * (int(pre[5]) - int(pre[3]) + 1)
                N_area = (int(cor[4]) - int(cor[2]) + 1) * (int(cor[5]) - int(cor[3]) + 1)

                iw = min(int(cor[4]), int(pre[4])) - max(int(cor[2]), int(pre[2])) + 1

                if (iw > 0):
                    ih = min(int(cor[5]), int(pre[5])) - max(int(cor[3]), int(pre[3])) + 1

                    if (ih > 0):
                        ua = (K_area + N_area) - iw*ih
                        ans = iw*ih / ua

                if (iw > 0 and ih > 0 and args.iou_threshold <= ans):
                    iou_list.append([cor[0], cor[1], pre[1], cor[6], pre[6], ans, pre[7]])

    # 昇順にソート
    iou_list.sort()

    # iou_list = [inference/directory/framename, cor_label, pre_label, cor_num, pre_num, iou, score]
    return iou_list


"""
---------------------------------------------------------------------------------------------------

IOUごとに評価分類（検出（正解＋推論）・不要）

---------------------------------------------------------------------------------------------------
"""
def classify_iou(iou_list):
    # ダブっている情報を削除
    for i, x in enumerate(iou_list[:]):
        for j, y in enumerate(iou_list[:]):
            if (x[0] == y[0] and x[3] != y[3] and x[4] == y[4]):
                # iouが小さい方を削除
                if (x[5] >= y[5]):
                    iou_list.remove(y)

    # 不要
    unnecessary = []

    # リストから"不要"を取り出す
    for i, x in enumerate(iou_list[:]):
        for j, y in enumerate(iou_list[:]):
            if (x[0] == y[0] and x[3] == y[3] and x[4] != y[4]):
                # iouが小さい方を"不要"に
                if (x[5] >= y[5]):
                    unnecessary.append(y)
                    iou_list.remove(y)

    # 検出（正解＋推論）
    detection = iou_list

    return detection, unnecessary


"""
---------------------------------------------------------------------------------------------------

評価ごとに数をカウント

---------------------------------------------------------------------------------------------------
"""
def count_evaluation_num(categories, pre_boxnum_list, cor_boxnum_list, detection, unnecessary):
    # 静的確保
    count_eval = [[0 for i in range(len(categories)*2 + 3)] for j in range(len(categories))]

    # 検出（正解＋推論）
    for i, x in enumerate(categories):
        for j, y in enumerate(categories):
            for obj in detection:
                if (x == obj[1] and y == obj[2]):
                    count_eval[i][1 + j] += 1

                if (x == obj[2] and y == obj[1]):
                    count_eval[i][1 + len(categories) + j] += 1

    # 不要
    for i, category in enumerate(categories):
        for obj in unnecessary:
            if (category == obj[2]):
                count_eval[i][2 + len(categories)*2] += 1

    # 過検出
    for i, box in enumerate(count_eval):
        pre_box_num  = 0

        for j in range(1 + len(categories), len(categories)*2 + 3):
            pre_box_num += count_eval[i][j]

        t = pre_boxnum_list[i][1] - pre_box_num
        count_eval[i][len(categories)*2 + 1] = t

    # 未検出
    for i, box in enumerate(count_eval):
        cor_box_num  = 0

        for j in range(1, len(categories) + 1):
            cor_box_num += count_eval[i][j]

        t = cor_boxnum_list[i][1] - cor_box_num
        count_eval[i][0] = t

    # count_eval = [未検出, 正解検出(1), ・・・, 正解検出(n), 推論検出(1), ・・・, 推論検出(n), 過検出, 不要]
    return count_eval


"""
---------------------------------------------------------------------------------------------------

平均IOUを計算

---------------------------------------------------------------------------------------------------
"""
def calculate_averageiou(categories, detection):
    avg_iou = []

    for i, category in enumerate(categories):
        div = 0
        sum_iou = 0

        for dct in detection:
            if (dct[1] == dct[2] and category == dct[1]):
                div += 1
                sum_iou += dct[5]

        # 平均を算出
        if (div == 0):
            avg = 0

        else:
            avg = '{:.2f}'.format(sum_iou/div)

        avg_iou.append(avg)

    return avg_iou

"""
---------------------------------------------------------------------------------------------------

再現率を計算

---------------------------------------------------------------------------------------------------
"""
def calculate_recall(count_eval):
    recalls= []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if j <= size:
                Sum += int(num)
        if Sum != 0:
            recall = int(box[1+i]) / Sum 
            recalls.append(recall)
        else:
            recalls.append(0)

    return recalls
#全足し合わせ
def calculate_recall_z(count_eval):
    recalls= []
    Sums = []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if j <= size:
                Sum += int(num)
        Sums.append(Sum)

    if Sum != 0:
        recall = (int(count_eval[0][1])+int(count_eval[1][2])+int(count_eval[2][3])) / (int(Sums[0])+int(Sums[1])+int(Sums[2])) 
        recalls.append(recall)
    else:
        recalls.append(0)

    return recalls
"""
---------------------------------------------------------------------------------------------------

適合率を計算

---------------------------------------------------------------------------------------------------
"""
def calculate_precisions(count_eval):
    precisions= []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if  size < j < (size+1) * 2:
                Sum += int(num)
        if Sum != 0:
            precision = int(box[size+1+i]) / Sum 
            precisions.append(precision)
        else:
            precisions.append(0)

    return precisions
#全足し合わせ
def calculate_precisions_z(count_eval):
    precisions= []
    Sums = []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if  size < j < (size+1) * 2:
                Sum += int(num)
        Sums.append(Sum)
    sumsum = int(Sums[0])+int(Sums[1])+int(Sums[2])
    if sumsum != 0:
        precision = (int(count_eval[0][size+1+0])+int(count_eval[1][size+1+1])+int(count_eval[2][size+1+2])) / (int(Sums[0])+int(Sums[1])+int(Sums[2]))  
        precisions.append(precision)
    else:
        precisions.append(0)

    return precisions
#F値
def calculate_fmeasure(recalls,precisions):
    fmeasure_list= []
    for i in range(len(recalls)):
        if  recalls[i] != 0 and precisions[i] != 0:
            fmeasure = 2 * recalls[i] * precisions[i] / ( recalls[i] + precisions[i] )
        else:
            fmeasure = 0
        fmeasure_list.append(fmeasure)
    
    return fmeasure_list



#csv
def write_csv(rotate,name):
    save_row = {}
    save_row2 = {}
    with open(os.path.join(args.pre_xml,'%s.csv' % name),'w') as f:
        fieldnames = ["name","recalls", "precisions"]
        writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(rotate.keys())
        length = len(rotate)

        for minn in np.arange(0.0,args.thres,0.1):
            for j in range(2):
                k =rotate[str(minn)][j]
                save_row[cate(j)] = k
            save_row['name'] = str(minn)
            writer.writerow(save_row)

def cate(j):
    if j==0:
        return 'recalls'
    if j==1:
        return 'precisions'
    if j==2:
        return 'greenhouse'

#スコア
def caluculate_score(pre_label_list,cor_label_list,categories,pre_boxnum_list,cor_boxnum_list):
    # IOUを計算
    iou_list = calculate_iou(pre_label_list, cor_label_list)
    # IOUごとに評価分類（検出（正解＋推論）・不要）
    detection, unnecessary = classify_iou(iou_list)
    #評価ごとに数をカウント
    count_eval = count_evaluation_num(categories, pre_boxnum_list, cor_boxnum_list, detection, unnecessary)
    # 平均IOUを計算
    avg_iou = calculate_averageiou(categories, detection)
    # 再現率を計算
    recalls = calculate_recall(count_eval)
    # 適合率を計算
    precisions = calculate_precisions(count_eval)
    #F値を計算
    fmeasure = calculate_fmeasure(recalls,precisions)

    return recalls,precisions,fmeasure


"""
---------------------------------------------------------------------------------------------------

メイン関数

---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    # 存在有無をチェック
    if not (os.path.isfile(args.category)):
        sys.exit('Not exist category text file ({})'.format(args.category))

    # カテゴリの読み込み
    with open(args.category, 'r') as f:
        categories = f.read().splitlines()
        #categories = ['labelname']

    # 存在有無をチェック
    if not (os.path.isdir(args.pre_xml)):
        sys.exit('Not exist xml file path ({})'.format(args.pre_xml))

    if not (os.path.isdir(args.cor_xml)):
        sys.exit('Not exist xml file path ({})'.format(args.cor_xml))
    # XMLファイルを取得
    cor_xml_list = glob.glob(os.path.join(args.cor_xml, '**', 'supervised', '*.xml'))
    # XMLファイルからラベルを取得
    cor_label_list = get_label(cor_xml_list, correct = True)
    # ボックス合計数を取得
    cor_boxnum_list = get_boxnum(cor_label_list, categories)
    rere = {}
    for s_threshold in np.arange(0.0,args.thres,0.1):
        s_s = "{0:.1f}".format(s_threshold)
        # XMLファイルを取得
        pre_xml_list = glob.glob(os.path.join(args.pre_xml,'draw-nms-marge%s' % s_s, '**','supervised','*.xml'))
        # XMLファイルからラベルを取得
        pre_label_list = get_label(pre_xml_list, correct = False)
        # ボックス合計数を取得
        pre_boxnum_list = get_boxnum(pre_label_list, categories)
        rr = []
        recalls,precisions,fmeasure = caluculate_score(pre_label_list,cor_label_list,categories,pre_boxnum_list,cor_boxnum_list)
        reca = (recalls[0]+recalls[1]+recalls[2])/3
        prec = (precisions[0]+precisions[1]+precisions[2])/3
        fmea = (fmeasure[0]+fmeasure[1]+fmeasure[2])/3
        rr.append(reca)
        rr.append(prec)
        rr.append(fmea)
        rere[str(s_threshold)] = rr

    write_csv(rere,args.name)
