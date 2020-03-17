import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outfilepath', type=str)
parser.add_argument('-f', '--inputfilepath', type=str, default='/home/hiroshi/Paddy_field/split_test/00/out/00')
parser.add_argument('-ff', '--test', type=str, default='../output/predict')
parser.add_argument('-s', '--search', type=str, default='max_speed_50')
args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'**', 'labelxml_rotate' ,'*.xml'))
        #print(xml_list)
        
        return xml_list

#value検索
def valuesearch(xml_list):
        name_list=[]
        count_list=[]
        for xml_path in xml_list:
                tree = ET.parse(xml_path)
                #ファイルの名前
                filename = xml_path.rsplit('/', 1)[1]
                newfilename = filename.split('.')
                #フォルダの名前
                foldername = xml_path.rsplit('/', 3)[1]
                #print(filename)
                #directory = xml_path.rsplit('/', 3)[1]
                #framename = xml_path.rsplit('/', 3)[3].rsplit('.', 1)[0]
                #dir_frame = os.path.join(directory, framename)
                
                # nameタグ
                names = tree.findall('object/category/type')
        
                #フォルダ名、ファイル名の変更
                # fileタグ
                tree.find('filename').text = newfilename[0]
                #書き込み
                tree.write(xml_path, encoding='UTF-8')
                 # folderタグ
                tree.find('folder').text = foldername
                tree.write(xml_path, encoding='UTF-8')
                '''
                for name in names:
                    name.text = 'agri_field'
                    tree.write(xml_path, encoding='UTF-8')
                '''
     
        if len(name_list) == 0:
                print('nothing')
        mycounter = Counter(count_list)
        print(mycounter)
        return name_list

def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        value_list = valuesearch(xml_list)
        print(value_list)
