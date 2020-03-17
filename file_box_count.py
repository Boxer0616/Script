import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter
import numpy as np
import cv2
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='/home/hiroshi/Chainer_Mask_R-CNN/dataset_mask/label')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test ,'*.xml'))
        #print(xml_list)
        xml_list = sorted(xml_list)
        
        return xml_list

#value検索
def valuesearch(xml_list):
        field_count = 0
        green_field_count = 0
        greenhouse_count = 0
        for xml_path in tqdm(xml_list):
                time.sleep(0.000001)
                tree = ET.parse(xml_path)
                #root
                root = tree.getroot()
                #object
                objec = root.findall('.//object')
                for obj in objec:
                        category = obj.find('./category/value')
                        if category.text == 'field':
                                field_count += 1
                        if category.text == 'green_field':
                                green_field_count += 1
                        if category.text == 'greenhouse':
                                greenhouse_count += 1

        return field_count,green_field_count,greenhouse_count




if __name__ == '__main__':
        xml_list = xmlopen()
        f_c,g_c,gh_c = valuesearch(xml_list)
        print('ファイル数')
        print(len(xml_list))
        print('植生なしほ場数')
        print(f_c)
        print('植生ありほ場数')
        print(g_c)
        print('ビニールハウス数')
        print(gh_c)
