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
import cv2
import time
from tqdm import tqdm
import copy

parser = argparse.ArgumentParser(description = 'Caluclate IOU and Detection rate')

parser.add_argument('-c', '--category', help = 'Path to category text file', type = str, default = './category.txt')
parser.add_argument('-px', '--pre_xml', help = 'Path to predict xml directory', type = str, default = './dataset/result/yolo/inference')
parser.add_argument('-cx', '--cor_xml', help = 'Path to correct xml directory', type = str, default = './dataset/val/validation')
parser.add_argument('-na', '--name',type = str, default = 'yolo')
parser.add_argument('-t', '--thres',type = float, default = 1.1)
parser.add_argument('-st', '--start_thres',type = float, default = 0.0)
parser.add_argument('-i', '--iou_threshold', help = 'Threshold of IoU', type = float, default = 0.3)

# action ---> オプション--✕✕をつけるとTrue、つけないとデフォルト値のFalse
parser.add_argument('-r', '--rotate_box',    action='store_true')
#再現率適合率をカテゴリ平均で計算するか合計で計算するか
parser.add_argument('-a', '--average',    action='store_true')

args = parser.parse_args()


"""
---------------------------------------------------------------------------------------------------

XMLファイルからラベルを取得

---------------------------------------------------------------------------------------------------
"""
def get_label(xml_list, categories, dirform=True, correct=False):
    label_list = []

    for xml_path in xml_list:
        tree = ET.parse(xml_path)

        framename = ''

        if (dirform):
            directory = xml_path.rsplit('/', 3)[1]
            framename = xml_path.rsplit('/', 3)[3].rsplit('.', 1)[0]
            framename = os.path.join(directory, framename)

        else:
            framename = xml_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        num = 0

        for object in tree.findall('object'):
            labelname = ''

            for cat in object.findall('category'):
                for cls in categories:
                    if (cls == cat.find('value').text):
                        labelname = cls

            if (object.find('bndbox').find('xmin') == None):
                sys.exit('Missmatch Box Type')

            xmin = int(object.find('bndbox').find('xmin').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymax = int(object.find('bndbox').find('ymax').text)

            if not (correct):
                score = object.find('score').text
                label_list.append([framename, labelname, xmin, ymin, xmax, ymax, num, score])

            else:
                label_list.append([framename, labelname, xmin, ymin, xmax, ymax, num])

            # １枚の画像に含まれるバウンディングボックス数をカウント
            num += 1

    # 昇順にソート
    label_list.sort()

    # label_list = [directory/framename or framename, labelname, x1, y1, x2, y2, num, (score)]
    return label_list

"""
---------------------------------------------------------------------------------------------------

XMLファイルからラベルを取得（回転ver.）

---------------------------------------------------------------------------------------------------
"""
def get_label_rotate(xml_list, categories, dirform=True, correct=False):
    label_list = []

    for xml_path in xml_list:
        tree = ET.parse(xml_path)

        framename = ''

        if (dirform):
            directory = xml_path.rsplit('/', 3)[1]
            framename = xml_path.rsplit('/', 3)[3].rsplit('.', 1)[0]
            framename = os.path.join(directory, framename)

        else:
            framename = xml_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        num = 0

        for object in tree.findall('object'):
            labelname = ''

            for cat in object.findall('category'):
                for cls in categories:
                    if (cls == cat.find('value').text):
                        labelname = cls

            if (object.find('bndbox').find('x0') == None):
                sys.exit('Missmatch Box Type')

            x0 = int(object.find('bndbox').find('x0').text)
            y0 = int(object.find('bndbox').find('y0').text)
            x1 = int(object.find('bndbox').find('x1').text)
            y1 = int(object.find('bndbox').find('y1').text)
            x2 = int(object.find('bndbox').find('x2').text)
            y2 = int(object.find('bndbox').find('y2').text)
            x3 = int(object.find('bndbox').find('x3').text)
            y3 = int(object.find('bndbox').find('y3').text)

            if not (correct):
                score = object.find('score').text
                label_list.append([framename, labelname, x0, y0, x1, y1, x2, y2, x3, y3, num, score])

            else:
                label_list.append([framename, labelname, x0, y0, x1, y1, x2, y2, x3, y3, num])

            # １枚の画像に含まれるバウンディングボックス数をカウント
            num += 1

    # 昇順にソート
    label_list.sort()

    # label_list = [directory/framename or framename, labelname, x1, y1, x2, y2, num, score]
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

    for cor in tqdm(correct_label_list):
        time.sleep(0.000001)
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

IOUを計算（回転ver.）

---------------------------------------------------------------------------------------------------
"""
def calculate_iou_rotate(predict_label_list, correct_label_list):
    iou_list = []

    for cor in tqdm(correct_label_list):
        time.sleep(0.000001)

        # 空の配列を生成
        cor_im = np.zeros((800, 800, 3), dtype=np.uint8)

        # ポリゴンの座標を指定
        pts = np.array([[0, 0], [0, 800], [800, 800], [800, 0]])
        dst = cv2.fillPoly(cor_im, pts =[pts], color=(100, 100, 100))

        contours = np.array([[cor[2], cor[3]], [cor[4], cor[5]], [cor[8], cor[9]], [cor[6], cor[7]]])
        cor_mask = cv2.fillPoly(dst, [contours], (255, 255, 255))

        cor_area = cv2.contourArea(contours)

        for pre in predict_label_list:
            if (cor[0] == pre[0]):
                # 空の配列を生成
                pre_im = np.zeros((800, 800, 3), dtype=np.uint8)

                # ポリゴンの座標を指定
                pts = np.array([[0, 0], [0, 800], [800, 800], [800, 0]])
                dst = cv2.fillPoly(pre_im, pts =[pts], color=(0, 0, 0))

                contours = np.array([[pre[2], pre[3]], [pre[4], pre[5]], [pre[6], pre[7]], [pre[8], pre[9]]])
                pre_mask = cv2.fillPoly(dst, [contours], (255, 255, 255))

                gray_cor_mask = cv2.cvtColor(cor_mask, cv2.COLOR_BGR2GRAY)
                gray_pre_mask = cv2.cvtColor(pre_mask, cv2.COLOR_BGR2GRAY)

                pre_area = cv2.contourArea(contours)

                overlap = (gray_cor_mask == gray_pre_mask).sum()
                ans = overlap / ((cor_area + pre_area) - overlap)

                if (args.iou_threshold <= ans):
                    iou_list.append([cor[0], cor[1], pre[1], cor[10], pre[10], ans, pre[11]])

    # iou_list = [framename, cor_label, pre_label, cor_num, pre_num, iou, score]
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
    recalls_z = []
    recalls = []
    Sums = []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if j <= size:
                Sum += int(num)
        ###
        if Sum !=0:
            #カテゴリ別再現率
            recall = int(box[1+i]) / Sum 
            recalls.append(recall)
        else:
            recalls.append(0)
        ###
        Sums.append(Sum)
    #正解率
    sumsum = int(Sums[0])+int(Sums[1])+int(Sums[2])
    if sumsum != 0:
        recall_z = (int(count_eval[0][1])+int(count_eval[1][2])+int(count_eval[2][3])) / (int(Sums[0])+int(Sums[1])+int(Sums[2])) 
        recalls_z.append(recall_z)
    else:
        recalls_z.append(0)

    return recalls_z,recalls
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
    precisions_z = []
    precisions = []
    Sums = []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if  size < j < (size+1) * 2:
                Sum += int(num)
        ##カテゴリ別
        if Sum != 0:
            precision = int(box[size+1+i]) / Sum 
            precisions.append(precision)
        else:
            precisions.append(0)
        ##
        Sums.append(Sum)
    ##正解率
    sumsum = int(Sums[0])+int(Sums[1])+int(Sums[2])
    if sumsum != 0:
        precision_z = (int(count_eval[0][size+1+0])+int(count_eval[1][size+1+1])+int(count_eval[2][size+1+2])) / (int(Sums[0])+int(Sums[1])+int(Sums[2]))  
        precisions_z.append(precision_z)
    else:
        precisions_z.append(0)

    return precisions_z,precisions
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
def write_csv(rotate,name,cate_num):
    save_row = {}
    save_row2 = {}
    with open(os.path.join(args.pre_xml,'%s.csv' % name),'w') as f:
        if cate_num == 3:
            fieldnames = ["name",'recalls','precisions','fmeasure']
        else:
            fieldnames = ["name",'field','green_field','greenhouse','Average']
        writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(rotate.keys())
        length = len(rotate)

        for minn in np.arange(args.start_thres,args.thres,0.1):
            for j in range(cate_num):
                k =rotate[str(minn)][j]
                save_row[fieldnames[j+1]] = k
            save_row['name'] = str(minn)
            writer.writerow(save_row)
#category
def cate_srele(cate_num):
    if cate_num == 3:
        return 'recalls','precisions','fmeasure'
    else:
        return 'field','green_field','greenhouse','Average'

def cate(j,cate1,cate2,cate3):
    if j==0:
        return cate1
    if j==1:
        return cate2
    if j==2:
        return cate3

#スコア計算
def caluculate_score(pre_label_list,cor_label_list,categories,pre_boxnum_list,cor_boxnum_list):
    # IOUを計算
    if (args.rotate_box):
        iou_list = calculate_iou_rotate(pre_label_list, cor_label_list)
    else:
        iou_list = calculate_iou(pre_label_list, cor_label_list)
    # IOUごとに評価分類（検出（正解＋推論）・不要）
    detection, unnecessary = classify_iou(iou_list)
    #評価ごとに数をカウント
    count_eval = count_evaluation_num(categories, pre_boxnum_list, cor_boxnum_list, detection, unnecessary)
    # 平均IOUを計算
    avg_iou = calculate_averageiou(categories, detection)
    #再現率、適合率、F値を計算
    recalls,precisions,fmeasure,recalls_z,precisions_z,fmeasure_z = culcurate_recalls_precisions_fmeasure(count_eval)

    return recalls,precisions,fmeasure,recalls_z,precisions_z,fmeasure_z

#再現率、適合率、F値を計算
def culcurate_recalls_precisions_fmeasure(count_eval):
    # 合計で再現率を計算
    recalls_z,recalls = calculate_recall_z(count_eval)
    # 合計で適合率を計算
    precisions_z,precisions = calculate_precisions_z(count_eval)
    #F値を計算
    fmeasure = calculate_fmeasure(recalls,precisions)
    fmeasure_z = calculate_fmeasure(recalls_z,precisions_z)

    return recalls,precisions,fmeasure,recalls_z,precisions_z,fmeasure_z

#ファイル取得
def get_path():
    # XMLファイルを取得
    cor_xml_list = glob.glob(os.path.join(args.cor_xml, '**', 'supervised', '*.xml'))
    pre_xml_list = glob.glob(os.path.join(args.pre_xml, '**', 'supervised', '*.xml'))

    return cor_xml_list,pre_xml_list

#スコアで分ける
def split_score(s_threshold,pre_label_list):
    new_label_list = []
    for label in pre_label_list:
        #print(label[11])
        if (args.rotate_box):
            if float(label[11]) >= s_threshold:
                new_label_list.append(label)
        else:
            if float(label[7]) >= s_threshold:
                new_label_list.append(label)
    
    return new_label_list

#再現率、適合率、f値平均
def mean_category_score(rere,recalls,precisions,fmeasure):      
    rr = []
    #カテゴリ平均計算
    reca = (recalls[0]+recalls[1]+recalls[2])/3
    #print('field%s' % recalls[0])
    #print(reca)
    prec = (precisions[0]+precisions[1]+precisions[2])/3
    fmea = (fmeasure[0]+fmeasure[1]+fmeasure[2])/3
    #格納
    rr.append(reca)
    rr.append(prec)
    rr.append(fmea)
    rere[str(s_threshold)] = rr
    
    return rr




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

    # 存在有無をチェック
    if not (os.path.isdir(args.pre_xml)):
        sys.exit('Not exist predict xml directory ({})'.format(args.pre_xml))

    if not (os.path.isdir(args.cor_xml)):
        sys.exit('Not exist correct xml directory ({})'.format(args.cor_xml))

    pre_xml_list = []
    pre_dir_bool = True

    # XMLファイルを取得
    if (len(glob.glob(os.path.join(args.pre_xml, '*.xml'))) != 0):
        pre_xml_list = glob.glob(os.path.join(args.pre_xml, '*.xml'))
        pre_dir_bool = False

    else:
        pre_xml_list = glob.glob(os.path.join(args.pre_xml, '**', '**', '*.xml'))

    cor_xml_list = []
    cor_dir_bool = True

    # XMLファイルを取得
    if (len(glob.glob(os.path.join(args.cor_xml, '*.xml'))) != 0):
        cor_xml_list = glob.glob(os.path.join(args.cor_xml, '*.xml'))
        cor_dir_bool = False

    else:
        cor_xml_list = glob.glob(os.path.join(args.cor_xml, '**', '**', '*.xml'))

    dir_bool = True

    if not (pre_dir_bool and cor_dir_bool):
        dir_bool = False

    pre_label_list = []
    cor_label_list = []

    # XMLファイルからラベルを取得
    if (args.rotate_box):
        pre_label_list = get_label_rotate(pre_xml_list, categories, dirform=dir_bool, correct = False)
        cor_label_list = get_label_rotate(cor_xml_list, categories, dirform=dir_bool, correct = True)

    else:
        pre_label_list = get_label(pre_xml_list, categories, dirform=dir_bool, correct = False)
        cor_label_list = get_label(cor_xml_list, categories, dirform=dir_bool, correct = True)

    # ボックス合計数を取得
    cor_boxnum_list = get_boxnum(cor_label_list, categories)
    rere_z = {}
    rere = {}
    recall_m = {}
    preci_m = {}
    fmea_m = {}
    for s_threshold in tqdm(np.arange(args.start_thres,args.thres,0.1)):
        time.sleep(0.000001)
        s_s = "{0:.1f}".format(s_threshold)
        #スコアで分ける
        new_pre_label_list = split_score(s_threshold,pre_label_list)
        # ボックス合計数を取得
        pre_boxnum_list = get_boxnum(new_pre_label_list, categories)
        rr_z = []
        recalls,precisions,fmeasure,recalls_z,precisions_z,fmeasure_z = caluculate_score(new_pre_label_list,cor_label_list,categories,pre_boxnum_list,cor_boxnum_list)
        rr_z.append(recalls_z[0])
        rr_z.append(precisions_z[0])
        rr_z.append(fmeasure_z[0])
        rere_z[str(s_threshold)] = rr_z
        recalls_list = []
        precisions_list = []
        fmeasure_list = []        
        rr = mean_category_score(rere,recalls,precisions,fmeasure)
        #カテゴリ別再現率、適合率、F値格納
        re_list = copy.copy(recalls)
        re_list.append(rr[0])
        pre_list =  copy.copy(precisions)
        pre_list.append(rr[1])
        fmea_list =  copy.copy(fmeasure)
        fmea_list.append(rr[2])
        recall_m[str(s_threshold)] = re_list
        preci_m[str(s_threshold)] = pre_list
        fmea_m[str(s_threshold)] = fmea_list

    write_csv(rere,args.name,3)
    write_csv(rere_z,args.name+'_'+'seikairitu',3)
    write_csv(recall_m,args.name+'_'+'recalls',4)
    write_csv(preci_m,args.name+'_'+'precisions',4)
    write_csv(fmea_m,args.name+'_'+'fmeasure',4)
