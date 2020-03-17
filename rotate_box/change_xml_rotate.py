
import os
import glob
import xml.etree.ElementTree as ET
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', default = './category2.txt')
parser.add_argument('-t', '--text_dir', default = './data/R2CNN/predict/labeltxt_rotate')
parser.add_argument('-s', '--score_threshold', default = 0.6)
args = parser.parse_args()


"""
--------------------------------------------------------------------------------------------

カテゴリを読み込み

--------------------------------------------------------------------------------------------
"""
def read_category():
    # カテゴリの読み込み
    with open(args.category, 'r') as f:
        CLASSES = f.read().splitlines()

    return CLASSES


"""
--------------------------------------------------------------------------------------------

txtファイルを読込

--------------------------------------------------------------------------------------------
"""
def read_txt(text_file):
    txt_list = []

    # 読み込み
    f = open(text_file)
    lines = f.readlines()
    f.close()

    for line in lines:
        data = line.rstrip().split(' ')
        txt_list.append(data)

    return txt_list


"""
--------------------------------------------------------------------------------------------

xmlファイルを作成

--------------------------------------------------------------------------------------------
"""
def create_xml(framename, new_txt_list):
    root   = ET.Element('annotation')
    filename = ET.SubElement(root, 'filename')
    filename.text = framename

    for d in new_txt_list:
        obj    = ET.SubElement(root, 'object')
        bndbox = ET.SubElement(obj, 'bndbox')

        x0 = ET.SubElement(bndbox, 'x0')
        y0 = ET.SubElement(bndbox, 'y0')
        x1 = ET.SubElement(bndbox, 'x1')
        y1 = ET.SubElement(bndbox, 'y1')
        x2 = ET.SubElement(bndbox, 'x2')
        y2 = ET.SubElement(bndbox, 'y2')
        x3 = ET.SubElement(bndbox, 'x3')
        y3 = ET.SubElement(bndbox, 'y3')

        x0.text = str(int(d[1]))
        y0.text = str(int(d[2]))
        x1.text = str(int(d[3]))
        y1.text = str(int(d[4]))
        x2.text = str(int(d[5]))
        y2.text = str(int(d[6]))
        x3.text = str(int(d[7]))
        y3.text = str(int(d[8]))

        score = ET.SubElement(obj, 'score')
        score.text = str(d[9])

        category = ET.SubElement(obj, 'category')
        value    = ET.SubElement(category, 'value')
        value.text = str(d[0])

    save_dir = os.path.join(args.text_dir.rsplit('/', 1)[0], 'labelxml_rotate0.6')

    # ディレクトリを作成
    if not (os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(save_dir, framename + '.xml'))

"""
--------------------------------------------------------------------------------------------

メイン関数

--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    CLASSES = read_category()

    text_file_list = glob.glob(os.path.join(args.text_dir, '*.txt'))
    text_file_list.sort()

    for text_file in text_file_list:
        framename = text_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
        txt_list = read_txt(text_file)

        new_txt_list = []
        for txt in txt_list:
            for i, cls in enumerate(CLASSES):
                if ((int(txt[5]) == i) and (float(txt[6]) >= args.score_threshold)):
                    b = cv2.boxPoints(((int(txt[0]), int(txt[1])), (int(txt[2]), int(txt[3])), float(txt[4])))
                    new_txt_list.append([cls, b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1], txt[6]])

        # xmlファイル作成
        create_xml(framename, new_txt_list)
