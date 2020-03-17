import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./outputmask')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'label' ,'*.xml'))
        #print(xml_list)
        
        return xml_list

#value検索
def valuesearch(xml_list):
        for xml_path in xml_list:
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
                                print(xml_path)
                                print(area)
                                root.remove(obj)

                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
        print('successe')
