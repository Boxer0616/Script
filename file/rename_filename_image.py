import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-tt', '--test', help = 'Path to predict xml directory', type = str, default = './outputmask')
args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'image', '*.jpg'))
        #print(xml_list)
        xml_list = sorted(xml_list)
        return xml_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#value検索
def valuesearch(xml_list):
        for xml_path in xml_list:
                directory,frame = path_to_name(xml_path)
                pp = frame.rsplit('_')
                tt = xml_path.rsplit('/')
                new_number = str(int(pp[4]))+ '_' +str(int(pp[5]))
                new_name = pp[0]+'_'+pp[1]+'_'+pp[2]+'_'+pp[3]+'_'+new_number
                new_path = os.path.join(tt[0],tt[1],tt[2],new_name+'.jpg')
                
                
                os.rename(xml_path,new_path)


def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
