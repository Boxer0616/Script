import sys
import argparse
import glob
import os
import numpy as np
import colorsys
import imghdr
import shutil
import random
from distutils.util import strtobool
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', default = './category.txt')
parser.add_argument('-i', '--input_dir',  default = './dataset/result/R2CNN/xml')
parser.add_argument('-ii', '--input_dir2',  default = './dataset/val/validation')
parser.add_argument('-o', '--out_dir',  default = './dataset/result/R2CNN/draws0.6')
args = parser.parse_args()

"""
--------------------------------------------------------------------------------------------

カテゴリを読み込み

--------------------------------------------------------------------------------------------
"""
def read_category():
    # カテゴリの読み込み
    with open(args.category, 'r') as f:
        CLASSES = f.read().splitlines()

    return CLASSES

"""
---------------------------------------------------------------------------------------------------

XMLファイルからラベルを取得

---------------------------------------------------------------------------------------------------
"""
def get_label(xml_path,n, CLASSES):
    label_list = []

    tree = ET.parse(xml_path)
    #filename = tree.find('filename').text
    filename = n
    for object in tree.findall('object'):
        cls_num = 0

        labelname = object.find('category').find('value').text

        for i, cls in enumerate(CLASSES):
            if (cls == labelname):
                cls_num = i

        x0 = object.find('bndbox').find('x0').text
        y0 = object.find('bndbox').find('y0').text
        x1 = object.find('bndbox').find('x1').text
        y1 = object.find('bndbox').find('y1').text
        x2 = object.find('bndbox').find('x2').text
        y2 = object.find('bndbox').find('y2').text
        x3 = object.find('bndbox').find('x3').text
        y3 = object.find('bndbox').find('y3').text

        #label_txt = xmin + ' ' + ymin + ' ' + xmin + ' ' + ymax + ' ' + xmax + ' ' + \
         #           ymin + ' ' + xmax + ' ' + ymax + ' ' + labelname + ' 1'
        score = object.find('score').text
        label_txt = x0 + ' ' + y0 + ' ' + x1 + ' ' + y1 + ' '+ x2 + ' ' + y2 + ' ' + x3 + ' ' + y3 + ' '+ labelname + ' ' + filename + ' ' + score + ' 1'
        label_list.append(label_txt)

    return label_list

"""
---------------------------------------------------------------------------------------------------

    インプット読み込み

---------------------------------------------------------------------------------------------------
"""
def read_imput(path_lst):
    predict_path = []
    for path in  path_lst:
        predict_path.append(path)

    return predict_path




    """
---------------------------------------------------------------------------------------------------

メイン関数

---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    CLASSES = read_category()
    label_list = []
    img_path_list = glob.glob(os.path.join(args.input_dir2 , '**','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir, '*.xml'))
    xml_path_list = sorted(xml_path_list)
    i = 0
    for n,img_path in enumerate(img_path_list):
        directory = img_path.rsplit('/', 3)[1]
        framename = img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        img = Image.open(img_path)
        img2 = cv2.imread(img_path)
        xml = get_label(xml_path_list[n], framename ,CLASSES)
        for xml_labe in xml:
            labe = xml_labe.split(' ')
            l = labe[9]
            xmin = int(labe[0])
            ymin = int(labe[1])
            xmax = int(labe[2])
            ymax = int(labe[3])
            if l==framename:
                if float(labe[10])>=0.6:
                    d = ImageDraw.Draw(img)
                    if labe[8] == 'green_field':
                        cv2.line(img2,(int(labe[0]), int(labe[1])), (int(labe[2]), int(labe[3])), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[2]), int(labe[3])), (int(labe[4]), int(labe[5])), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[4]), int(labe[5])), (int(labe[6]), int(labe[7])), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[6]), int(labe[7])), (int(labe[0]), int(labe[1])), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    if labe[8] == 'field':
                        cv2.line(img2,(int(labe[0]), int(labe[1])), (int(labe[2]), int(labe[3])), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[2]), int(labe[3])), (int(labe[4]), int(labe[5])), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[4]), int(labe[5])), (int(labe[6]), int(labe[7])), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[6]), int(labe[7])), (int(labe[0]), int(labe[1])), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    if labe[8] == 'greenhouse':
                        cv2.line(img2,(int(labe[0]), int(labe[1])), (int(labe[2]), int(labe[3])), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[2]), int(labe[3])), (int(labe[4]), int(labe[5])), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[4]), int(labe[5])), (int(labe[6]), int(labe[7])), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(img2,(int(labe[6]), int(labe[7])), (int(labe[0]), int(labe[1])), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        #d.rectangle([(xmin, ymin), (xmax, ymax)], outline='blue', width=1)
                        #d.text((xmin, ymin),'greenhouse',fill='blue')
                    #$cv2.rectangle(img,l[],pt2,color,thickness)
                    #img.show()
            #img.save(args.out_dir + '/' + framename + '.jpg', quality=95)
        # ディレクトリ作成
        if not (os.path.exists(args.out_dir)):
            os.makedirs(args.out_dir)
        cv2.imwrite(args.out_dir + '/' + framename + '.jpg',img2)
