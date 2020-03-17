
import os
import glob
import xml.etree.ElementTree as ET
import argparse
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--text_path', default = '../output/predict-final0.3-io0.5')
parser.add_argument('-s', '--score_threshold', default = 0.6)
args = parser.parse_args()


"""
--------------------------------------------------------------------------------------------

txtファイルを読込

--------------------------------------------------------------------------------------------
"""
def read_txt(txt_list, text_file, category):
    # 読み込み
    f = open(text_file)
    lines = f.readlines()
    f.close()

    for line in lines:
        data = line.rstrip().split(' ')
        data.append(category)
        txt_list.append(data)

def search_file(path):
    files = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
            filename = filename.strip('.xml')
            files.append(filename)
    return files

def new_or_write(file,framename):
    for ff in file:
        if framename == ff:
            return True
    return False


"""
--------------------------------------------------------------------------------------------

xmlファイルを作成

--------------------------------------------------------------------------------------------
"""
def create_xml(data):
    name = 0
    for data in data_list:
        directory = data[0].rsplit('/', 3)[1]
        framename = data[0].rsplit('/', 3)[3].rsplit('.', 1)[0]
        dir_name = data[0].strip(framename)
        dir_name = dir_name.rstrip('/')
        file = search_file(dir_name)

        a = new_or_write(file,framename)
        if a == False:
            root   = ET.Element('annotation')
            folder = ET.SubElement(root, 'folder')
            folder.text = str(data[0].rsplit('/', 2)[1])
            filename = ET.SubElement(root, 'filename')
            filename.text = str(data[0].rsplit('/', 2)[2])
            
            for d in data[1:]:
                obj    = ET.SubElement(root, 'object')
                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')

                xmin.text = str(int(float(d[1])))
                ymin.text = str(int(float(d[2])))
                xmax.text = str(int(float(d[3])))
                ymax.text = str(int(float(d[4])))

                score = ET.SubElement(obj, 'score')
                score.text = str(round(float(d[0]), 2))

                category = ET.SubElement(obj, 'category')
                value    = ET.SubElement(category, 'value')
                value.text = str(d[5])
            #ツリー反映
            tree = ET.ElementTree(root)
            #改行
            indent(root)
            #書き込み
            tree.write(data[0] + '.xml')
            name=data[0]
        else:
            tree = ET.parse(data[0]+'.xml')
            # ツリーを取得
            root = tree.getroot()
            folder.text = str(data[0].rsplit('/', 2)[1])
            filename.text = str(data[0].rsplit('/', 2)[2])
            for d in data[1:]:
                obj    = ET.SubElement(root, 'object')
                bndbox = ET.SubElement(obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')

                xmin.text = str(int(float(d[1])))
                ymin.text = str(int(float(d[2])))
                xmax.text = str(int(float(d[3])))
                ymax.text = str(int(float(d[4])))

                score = ET.SubElement(obj, 'score')
                score.text = str(round(float(d[0]), 2))

                category = ET.SubElement(obj, 'category')
                value    = ET.SubElement(category, 'value')
                value.text = str(d[5])
            #ツリー反映
            tree = ET.ElementTree(root)
            #改行
            indent(root)
            #書き込み
            tree.write(data[0] + '.xml')
            name=data[0]

#xml書き込み改行
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
"""
--------------------------------------------------------------------------------------------

メイン関数

--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    text_file_list = glob.glob(os.path.join(args.text_path, '*.txt'))

    txt_list = []
    for i, text_file in enumerate(text_file_list):
        category = text_file.split('_', 3)[3].rsplit('.', 1)[0]

        # txtファイル読込
        read_txt(txt_list, text_file, category)

    # 画像ごとに整理
    data_list = []
    for i, data in enumerate(tqdm(txt_list)):
        time.sleep(0.000001)
        # しきい値処理
        if ((float(data[2]) >= float(args.score_threshold)) and ('nan' not in data) and ('-inf' not in data)):
            output_dir  = os.path.join(args.text_path, 'inference0.6', data[0].rsplit('/', 1)[1],'supervised')
            output_path = os.path.join(output_dir, data[1])

            #ディレクトリを作成
            if not (os.path.isdir(output_dir)):
                os.makedirs(output_dir)

            if (i == 0):
                data_list.append([output_path, [data[2], data[3], data[4], data[5], data[6], data[7]]])

            elif (output_dir in [d[0] for d in data_list]):
                index = [d[0] for d in data_list].index(output_path)
                data_list[index].append([data[2], data[3], data[4], data[5], data[6], data[7]])

            else:
                data_list.append([output_path, [data[2], data[3], data[4], data[5], data[6], data[7]]])

    # xmlファイル作成
    #for data in data_list:
    data_list = sorted(data_list)
    create_xml(data_list)
