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
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './output-False-nms/sfam800/draw-nms-0.6')
parser.add_argument('-ii', '--input_dir2',  default = './output-False-nms/sfam320-512')
parser.add_argument('-o', '--out_dir',  default = './output-False-nms/sfam320-512-800')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir,'**','supervised','*.xml'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir2, '**' ,'supervised' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename


#xmlに書き込みと保存
def write_xml(xml_path_list,xml_path_list2):
    cccc = 0
    for xc,xml_path in enumerate(xml_path_list):
        for xc,xml_path2 in enumerate(xml_path_list2):
            directory,framename = path_to_name(xml_path)
            directory2,framename2 = path_to_name(xml_path2)
            if framename==framename2:
                tree = ET.parse(xml_path)
                tree2 = ET.parse(xml_path2)
                # ツリーを取得
                root = tree.getroot()
                root2 = tree2.getroot()
                filename = tree.find('filename').text
                #directory = tree.find('folder').text
                directory = xml_path.rsplit('/', 3)[1]
                for bc,obje in enumerate(tree2.findall('object')): 
                    bdb=[]
                    objc = ET.SubElement(root, 'object')
                    bndbox = ET.SubElement(objc, 'bndbox')
                    scrr = ET.SubElement(objc, 'score')
                    scrr.text = obje.find('./score').text
                    xmin= ET.SubElement(bndbox, 'xmin')
                    ymin = ET.SubElement(bndbox, 'ymin')
                    xmax= ET.SubElement(bndbox, 'xmax')
                    ymax = ET.SubElement(bndbox, 'ymax')
                    
                    xmin.text = obje.find('./bndbox/xmin').text
                    ymin.text = obje.find('./bndbox/ymin').text
                    xmax.text = obje.find('./bndbox/xmax').text
                    ymax.text = obje.find('./bndbox/ymax').text
            
                    # category
                    category = ET.SubElement(objc, 'category')
                    value = ET.SubElement(category, 'value')
                    value.text = obje.find('./category/value').text


                # ツリー反映
                tree = ET.ElementTree(root)
                #改行
                indent(root)
                #ディレクトリ作成
                create_dir(directory)
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
    if not (os.path.isdir(os.path.join(args.out_dir, directory,'supervised'))):
        os.makedirs(os.path.join(args.out_dir, directory,'supervised'))




if __name__ == '__main__':
    #データのパス
    xml_path_list,xml_path_list2 = import_data()
    #xmlに書き込み
    write_xml(xml_path_list,xml_path_list2)
