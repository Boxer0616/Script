import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./dataset/yolo')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        xml_list = glob.glob(os.path.join(args.test, '**' ,'supervised', '*.xml'))
        #xml_list = glob.glob(os.path.join(args.test,'label' ,'*.xml'))
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
                        category = obj.findall('category')
                        if len(category) !=1:
                                obj.remove(category[1])
                                #print(xml_path)
                                count += 1
                tree.write(xml_path, encoding='UTF-8')
        print(count)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
        print('successe')
