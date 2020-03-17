import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./out')

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
                #folder,filaename,path,source削除,
                folde = root.find('folder')
                fname = root.find('filename')
                path = root.find('path')
                source = root.find('source')
                root.remove(folde)
                root.remove(fname)
                root.remove(path)
                root.remove(source)
                #porygon削除、categoryのtype,language削除
                objec = root.findall('.//object')
                for obj in objec:
                        category = obj.find('category')
                        typp = category.find('type')
                        language = category.find('language')
                        polygon = obj.find('polygon')
                        category.remove(typp)
                        category.remove(language)
                        obj.remove(polygon)
                        
                tree.write(xml_path)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
