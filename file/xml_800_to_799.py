import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./dataset/mask07')

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
                        xmin = obj.find('./bndbox/xmin')
                        ymin = obj.find('./bndbox/ymin')
                        xmax = obj.find('./bndbox/xmax')
                        ymax = obj.find('./bndbox/ymax')
                        if int(xmin.text) >= 800:
                                xmin.text = '799'
                                print(xml_path)
                        if int(ymin.text) >= 800:
                                ymin.text = '799'
                                print(xml_path)
                        if int(xmax.text) >= 800:
                                xmax.text = '799'
                                print(xml_path)
                        if int(ymax.text) >= 800:
                                ymax.text = '799'
                                print(xml_path)
                        
                        if int(xmin.text) < 0:
                                xmin.text = '0'
                                print(xml_path)
                        if int(ymin.text) < 0:
                                ymin.text = '0'
                                print(xml_path)
                        if int(xmax.text) < 0:
                                xmax.text = '0'
                                print(xml_path)
                        if int(ymax.text) < 0:
                                ymax.text = '0'
                                print(xml_path)
                        #polygon
                        point = obj.findall('./polygon/point')
                        for poin in point:
                                point_x = poin.find('x')
                                point_y = poin.find('y')
                                if int(point_x.text) >= 800:
                                        point_x.text = '799'
                                        print(xml_path)
                                if int(point_y.text) >= 800:
                                        point_y.text = '799'
                                        print(xml_path)
                                if int(point_x.text) < 0:
                                        point_x.text = '0'
                                        print(xml_path)
                                if int(point_y.text) < 0:
                                        point_y.text = '0'
                                        print(xml_path)

                        tree.write(xml_path)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
        print('successe')
