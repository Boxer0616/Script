import glob
import random
import os
import xml.etree.ElementTree as ET
from PIL import Image
import copy
import argparse
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './outputmask')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir,'image','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir,'label','*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

def remove(anno, img, xml):

    ann = copy.deepcopy(anno)

    objs = ann.findall('object')

    if len(objs)==0:
        os.remove(xml)
        os.remove(img)
        print(ann)
        print(img)




def remove_file():
    img_path_list,xml_path_list = import_data()
    print('imgファイル数')
    print(len(img_path_list))
    print('xmlファイル数')
    print(len(xml_path_list))  
    if len(img_path_list)!=len(xml_path_list):
        print('ファイル数違う')
        sys.exit()
    for i,ann in enumerate(xml_path_list):
        fname = os.path.basename(ann)
        xml = ET.parse(ann)
        objs = xml.findall('object')
        if len(objs) == 0:
            os.remove(ann)
            os.remove(img_path_list[i])
    img_path_list,xml_path_list = import_data()
    print('imgファイル数削除後')
    print(len(img_path_list))
    print('xmlファイル数削除後')
    print(len(xml_path_list))  


if __name__ == '__main__':

    # sakujyo
    remove_file()
