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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', default = './category.txt')
parser.add_argument('-i', '--input_dir',  default = '/home/hiroshi/Dataset/folda-change/20191011矩形最終/remove-edge-val')
parser.add_argument('-ii', '--input_dir2',  default = './mike')
parser.add_argument('-o', '--out_dir',  default = './draw/sfam800-mike')
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
def get_label(xml_path, CLASSES):
    label_list = []

    tree = ET.parse(xml_path)
    filename = tree.find('filename').text

    for object in tree.findall('object'):
        cls_num = 0

        labelname = object.find('category').find('value').text

        for i, cls in enumerate(CLASSES):
            if (cls == labelname):
                cls_num = i

        xmin = object.find('bndbox').find('xmin').text
        ymin = object.find('bndbox').find('ymin').text
        xmax = object.find('bndbox').find('xmax').text
        ymax = object.find('bndbox').find('ymax').text

        #label_txt = xmin + ' ' + ymin + ' ' + xmin + ' ' + ymax + ' ' + xmax + ' ' + \
         #           ymin + ' ' + xmax + ' ' + ymax + ' ' + labelname + ' 1'
        label_txt = xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax + ' ' + labelname + ' ' + filename + ' 1'
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
    img_path_list = glob.glob(os.path.join(args.input_dir,'**','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir2,'**', 'supervised','*.xml'))
    xml_path_list = sorted(xml_path_list)
    i = 0
    for img_path in img_path_list:
        directory = img_path.rsplit('/', 3)[1]
        framename = img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        img = Image.open(img_path)
        for xml_path in xml_path_list:
            for label in get_label(xml_path, CLASSES):
                labe = label.split(' ')
                l = labe[5]
                xmin = int(labe[0])
                ymin = int(labe[1])
                xmax = int(labe[2])
                ymax = int(labe[3])
                if l==framename:
                    d = ImageDraw.Draw(img)
                    if labe[4] == 'green_field':
                        d.rectangle([(xmin, ymin), (xmax, ymax)], outline='lawngreen', width=1)
                        d.text((xmin, ymin),'green_field',fill='lawngreen')
                    if labe[4] == 'field':
                        d.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=1)
                        d.text((xmin, ymin),'field',fill='red')
                    if labe[4] == 'greenhouse':
                        d.rectangle([(xmin, ymin), (xmax, ymax)], outline='blue', width=1)
                        d.text((xmin, ymin),'greenhouse',fill='blue')
                    
                    #img.show()
        img.save(args.out_dir + '/' + framename + '.jpg', quality=95)
    a=0
