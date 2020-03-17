import sys
import argparse
import glob
import os
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from lxml import etree
import xml.dom.minidom as md
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './data/predict512/marge0.1')
parser.add_argument('-o', '--out_dir',  default = './data/predict512/margenms0.1')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir, 'img' ,'*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir,'**' ,'supervised','*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename



#xmlから矩形を取得
def get_bbox(xml_path):         
    sc_m = []
    nn_m = []
    cc_m = []
    obj_m = []
    for n, xml in enumerate(xml_path):
        tree = etree.parse(xml)
        filename = tree.find('filename').text
        objs = tree.xpath('//object')
        objj = []
        sc = []
        nn = []
        cc = []
        #オブジェクトごと
        for obj in objs:
            bdbs = obj.xpath('bndbox') 
            #bdbを取得
            for bdb in bdbs:
                objj.append([int(bdb.xpath('.//xmin')[0].text),
                             int(bdb.xpath('.//ymin')[0].text),
                             int(bdb.xpath('.//xmax')[0].text),
                             int(bdb.xpath('.//ymax')[0].text)])
                nn.append(filename)
            #スコアの取得
            scoree = obj.xpath('score')
            for scor in scoree:
                sc.append(float(scor.text))
            #カテゴリの取得
            cate = obj.xpath('.//category/value')
            for scr in cate:
                cc.append(scr.text)
        sc_m.append(sc)
        nn_m.append(nn)
        cc_m.append(cc)
        arr_obj = np.array(objj)
        obj_m.append(arr_obj)
    return obj_m,sc_m,nn_m,cc_m

#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory,'supervised'))):
        os.makedirs(os.path.join(args.out_dir, directory,'supervised'))

#xmlファイル作成
def create_xml(obj,scores,cate,namee):
    nn = namee[0].rsplit("_")
    file = nn[0] + '_' + nn[1] + '_' + nn[2] + '_' + nn[3]
    directory = nn[0] + '_' + nn[1]

    root   = ET.Element('annotation')
    # folderを追加する 
    folder = ET.SubElement(root, 'folder') 
    filename = ET.SubElement(root, 'filename')
    # pathを追加する 
    path = ET.SubElement(root, 'path')
    path.text = 'aa'
    # sourceを追加する 
    soource = ET.SubElement(root, 'soource')
    database = ET.SubElement(soource, 'database')
    # sizeを追加する 
    size = ET.SubElement(root, 'size') 
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    # segmentedを追加する 
    segmented = ET.SubElement(root, 'segmented')

    folder.text = directory
    filename.text = file
    database.text = 'Unknown'
    width.text = '2560'
    height.text = '2560'
    depth.text = '0'
    segmented.text = '0'

    for i,dd in enumerate(obj):
        # object 
        objc = ET.SubElement(root, 'object')
        pos = ET.SubElement(objc, 'pos')
        pos.text = 'Unspecified'
        truncated = ET.SubElement(objc, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(objc, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(objc, 'bndbox')
        scrr = ET.SubElement(objc, 'score')
        scrr.text = str(scores[i])
        xmin= ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax= ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        
        xmin.text = str(dd[0])
        ymin.text = str(dd[1])
        xmax.text = str(dd[2])
        ymax.text = str(dd[3])
   
        # category
        category = ET.SubElement(objc, 'category')
        value = ET.SubElement(category, 'value')
        value.text = str(cate[i])

    #tree = ET.ElementTree(root)
    #tree.write(xml_path)
    # ツリー反映
    tree = ET.ElementTree(root)
    #改行
    indent(root)
    create_dir(directory)
    #xml保存
    tree.write(os.path.join(args.out_dir, directory) + '/' + 'supervised' + '/'+ file  + '.xml', encoding="utf-8", xml_declaration=True)

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

def non_max_suppression(boxes, scores, cate,overlap_thresh):
    '''Non Maximum Suppression (NMS) を行う。

    Args:
        boxes     : (N, 4) の numpy 配列。矩形の一覧。
        overlap_thresh: [0, 1] の実数。閾値。

    Returns:
        boxes : (M, 4) の numpy 配列。Non Maximum Suppression により残った矩形の一覧。
    '''
    if boxes.size == 0:
        return []
    if len(boxes) <= 1:
        return boxes, scores, cate
    # float 型に変換する。
    boxes = boxes.astype("float")
    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(scores)  # スコアを降順にソートしたインデックス一覧
    selected = []  # NMS により選択されたインデックス一覧

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # indices は降順にソートされているので、一番最後の要素の値 (インデックス) が
        # 残っている中で最もスコアが高い。
        last = len(indices) - 1
        
        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        overlap = (i_w * i_h) / area[remaining_indices]

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # 選択された短形の一覧を返す。
    scores = np.array(scores)
    cate = np.array(cate)
    return boxes[selected].astype("int"),scores[selected],cate[selected]


if __name__ == '__main__':
    #ファイルパスの取得
    img_path_list,xml_path_list = import_data()
    #矩形とスコアを取得
    obje,scor,nn,cc = get_bbox(xml_path_list)
    obje = [i for i in obje if len(i)!=0]
    scor = [i for i in scor if len(i)!=0]
    nn = [i for i in nn if len(i)!=0]
    cc = [i for i in cc if len(i)!=0]

    for pp,obb in enumerate(obje):
        #nms
        box,sccc,catee = non_max_suppression(obb, scor[pp],cc[pp], 0.8)
        #xml作成
        create_xml(box,sccc,catee,nn[pp])
