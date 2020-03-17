import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter
import xml.dom.minidom as md

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./out')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'label' ,'*.xml'))
        #print(xml_list)
        
        return xml_list
def writeXml(xmlRoot, path):
    
    encode = "utf-8"
    
    xmlFile = open(path, "w")
    
    document = md.parseString(ET.tostring(xmlRoot, encode))
    
    document.writexml(
        xmlFile,
        encoding = encode,
        newl = "",
        indent = "",
        addindent = "\t"
    )

#value検索
def valuesearch(xml_list):
        for xml_path in xml_list:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                #フォルダ名、ファイル名の変更
                # fileタグ
                root.find('.//size/width').text = '800'
                #書き込み
                tree.write(xml_path)
                 # folderタグ
                root.find('.//size/height').text = '800'
                tree = ET.ElementTree(root)
                tree.write(xml_path)
                writeXml(root,xml_path)

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
