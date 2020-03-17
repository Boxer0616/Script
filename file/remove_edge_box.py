import sys
import argparse
import glob
import os
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from lxml import etree
import xml.dom.minidom as md
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = '../../rotate_bdb/output/tratestclop')
parser.add_argument('-o', '--out_dir',  default = './output')
parser.add_argument('-s', '--size',  default = '800')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir,'**','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir,'**','supervised','*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#クリップ後の画像端矩形削除
def remove_edge_box(xy_list,root,obj,filename):
    x = filename.rsplit('_', 1)[1]
    y = filename.rsplit('_', 2)[1]
    for lis in xy_list:
        if lis['xmin']<=2 or lis['xmax']>=(int(args.size))-1 or lis['ymin']<=2 or lis['ymax']>=(int(args.size))-1:
            root.remove(obj)
            break


#クリップ後の画像端矩形削除(0000、1760のときクロップされない方向は残す)
def remove_edge_boxv2(xy_list,root,obj,filename):
    x = filename.rsplit('_', 1)[1]
    y = filename.rsplit('_', 2)[1]
    for lis in xy_list:
        #if lis['xmin']<=0 or lis['xmax']>=(int(args.size)-1) or lis['ymin']<=0 or lis['ymax']>=(int(args.size)-1):
        if int(x) == 0:
            if lis['xmin'] <= 0 and lis['ymax']<(int(args.size)):
                continue
            else:
                if lis['xmax']>=(int(args.size)-1) or lis['ymax']>=(int(args.size)):
                    root.remove(obj)
                    break
        else:
            if int(x)+int(args.size) >= 2560:
                if lis['xmax']>=(int(args.size)):
                    continue
                else:
                    if lis['xmin']<=0:
                        root.remove(obj)
                        break
            else:
                if lis['xmin']<=0 or lis['xmax']>=(int(args.size)):
                    root.remove(obj)
                    break
        
        if int(y) == 0:
            if lis['ymin'] <= 0:
                continue
            else:
                if lis['ymax']>=(int(args.size)):
                    root.remove(obj)
                    break
        else:
            if int(y)+int(args.size) >= 2560:
                if lis['ymax']>=(int(args.size)):
                    continue
                else:
                    if lis['ymin']<=0:
                        root.remove(obj)
                        break
            else:
                if lis['ymin']<=0 or lis['ymax']>=(int(args.size)):
                    root.remove(obj)
                    break



#xmlに書き込みと保存
def write_xml(xml_path_list):
    for xc,xml_path in enumerate(xml_path_list):
        tree = ET.parse(xml_path)
        # ツリーを取得
        root = tree.getroot()
        filename = tree.find('filename').text
        directory = tree.find('folder').text
        for bc,obje in enumerate(tree.findall('object')): 
            bdb=[]
            #ノード読み込み
            bndb = obje.find('bndbox')
            #obje.insert(4, obje[-1])
            bdb.append({'xmin':int(bndb.find('xmin').text),
                        'ymin':int(bndb.find('ymin').text),
                        'xmax':int(bndb.find('xmax').text),
                        'ymax':int(bndb.find('ymax').text)})    
            remove_edge_box(bdb,root,obje,filename)

        # ツリー反映
        tree = ET.ElementTree(root)
        #改行
        indent(root)
        #ディレクトリ作成
        create_dir(directory+'/supervised')
        #xml保存
        tree.write(os.path.join(args.out_dir, directory)+ '/' +'supervised'+ '/' +filename+'.xml', encoding="utf-8", xml_declaration=True)

#xml書き込み改行
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory))):
        os.makedirs(os.path.join(args.out_dir, directory))

#画像コピー
def picture_copy(img_path_list):
    for img_path in img_path_list:
        img_data = cv2.imread(img_path)
        directory = img_path.rsplit('/', 2)[1]
        framename = img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        save_path = os.path.join(args.out_dir, directory, framename+'.jpg')
        cv2.imwrite(save_path,img_data)

if __name__ == '__main__':
    #データのパス
    img_path_list,xml_path_list = import_data()
    #xmlに書き込み
    write_xml(xml_path_list)
    #gazoucopy
    picture_copy(img_path_list)