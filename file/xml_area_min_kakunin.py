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
parser.add_argument('-ff', '--test', type=str, default='./dataset/yolo')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'**','supervised' ,'*.xml'))
        #print(xml_list)
        
        return xml_list

#value検索
def valuesearch(xml_list):
        count = 0
        for xml_path in tqdm(xml_list):
                time.sleep(0.000001)
                tree = ET.parse(xml_path)
                #root
                root = tree.getroot()
                #porygon削除、categoryのtype,language削除
                objec = root.findall('.//object')
                for obj in objec:
                        polys = []
                        polygons = []
                        #polygon
                        point = obj.findall('./polygon/point')
                        for poin in point:
                                x = int(poin.find('./x').text)
                                y = int(poin.find('./y').text)
                                polys.append((x,y))

                        polygons.append(polys)
                        #print(polygons)
                        area = cv2.contourArea(np.array(polygons))
                        if area == 0:
                                #print(xml_path)
                                #print(area)
                                count += 1
                                root.remove(obj)

                #tree.write(xml_path, encoding="utf-8", xml_declaration=False)
        print(count)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
        print('successe')
