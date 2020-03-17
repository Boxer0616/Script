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
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './dataset/20191011_形状最終/test')
parser.add_argument('-o', '--out_dir',  default = './output')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir,'**','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir,'**','supervised','*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 2)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#矩形とぽりごんを取得
def get_bbox_polygon(xml_path_list):
    objects = []
    polygons = [] 
    obj_cnt = []
    cnt = 0
    for xml_path in xml_path_list:
    
        tree = etree.parse(xml_path)
        filename = tree.find('filename').text
        obj_cnt.append(cnt)
        cnt = 0
        objs = tree.xpath('//object')

        for obj in objs:
            polys = []
            bdbs = obj.xpath('bndbox') 
            for bdb in bdbs:
                objects.append({
                'xmin' : int(bdb.xpath('.//xmin')[0].text),
                'xmax' : int(bdb.xpath('.//xmax')[0].text),
                'ymin' : int(bdb.xpath('.//ymin')[0].text),
                'ymax' : int(bdb.xpath('.//ymax')[0].text)
                })

            points = obj.xpath('.//polygon/point')
            for p in points:
                x = int(p.xpath('.//x')[0].text)
                y = int(p.xpath('.//y')[0].text)
                polys.append([x,y])
            if polys==[]:
                x = 0
                y = 0
                polys.append([x,y])
            polygons.append(polys)

    del obj_cnt[0]
    return objects,polygons

#xmlに書き込みと保存
def write_xml(xml_path_list,b,):
    count = 0
    for xml_path in xml_path_list:
        x = [''] * 4
        y = [''] * 4
        tree = ET.parse(xml_path)
        # ツリーを取得
        root = tree.getroot()
        filename = tree.find('filename').text
        dire = tree.find('folder').text
        for bndbox in tree.findall('object/bndbox'): 
            #xminとか削除
            xmin_etc_remove(bndbox,'xmin')
            xmin_etc_remove(bndbox,'ymin')
            xmin_etc_remove(bndbox,'xmax')
            xmin_etc_remove(bndbox,'ymax')
            xmin_etc_remove(bndbox,'x0')
            xmin_etc_remove(bndbox,'y0')
            xmin_etc_remove(bndbox,'x1')
            xmin_etc_remove(bndbox,'y1')
            xmin_etc_remove(bndbox,'x2')
            xmin_etc_remove(bndbox,'y2')
            xmin_etc_remove(bndbox,'x3')
            xmin_etc_remove(bndbox,'y3')
            #回転矩形の座標書き込み
            for n in range(4):
                x[n] = ET.SubElement(bndbox, 'x'+str(n))
                y[n] = ET.SubElement(bndbox, 'y'+str(n))
            for i in range(4):
                if b[count][i][0] < 0:
                    x[i].text = '0'
                else:
                    x[i].text = str(b[count][i][0])
                if b[count][i][1] < 0:
                    y[i].text = '0'
                else:
                    y[i].text = str(b[count][i][1])
            count += 1

        # ツリー反映
        tree = ET.ElementTree(root)
        #改行
        indent(root)
        output_dir_xml = os.path.join(args.out_dir,dire,'supervised')
        # ディレクトリ作成
        if not (os.path.exists(output_dir_xml)):
            os.makedirs(output_dir_xml)       
        #xml保存
        tree.write(output_dir_xml+ '/' +filename+'.xml', encoding="utf-8", xml_declaration=True)

#xmin,ymin,xma,ymax削除
def xmin_etc_remove(bndbox,remove_name):
    for re_neme in bndbox.findall(remove_name):
        bndbox.remove(re_neme)

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


#輪郭から回転矩形を取得
def get_rotate_bdb(xml_path_list,poly):
    rotate_bbox = []
    for i, cnt in enumerate(poly):
        #if len(cnt)!=0:
        cnt = np.array(cnt)
        # 輪郭に外接する回転した長方形を取得する。
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect
        print('bounding box of contour {} => '
            'center: ({:.2f}, {:.2f}), size: ({:.2f}, {:.2f}), angle: {:.2f}'.format(
            i, cx, cy, width, height, angle))
        # 回転した長方形の4点の座標を取得する。
        rect_points = cv2.boxPoints(rect)
        rect_points = np.array(rect_points,dtype=np.int64)
        rotate_bbox.append(rect_points)
        
    return rotate_bbox

if __name__ == '__main__':
    #データのパス
    img_path_list,xml_path_list = import_data()
    #矩形とポリゴンを取得
    obj,poly = get_bbox_polygon(xml_path_list)
    #回転矩形を取得
    rotate_bdb = get_rotate_bdb(xml_path_list,poly)
    #xmlに書き込み
    write_xml(xml_path_list,rotate_bdb)
    #画像コピー
    for out_path in img_path_list:
        directory2,framename2 = path_to_name(out_path)
        output_dir_img = os.path.join(args.out_dir,directory2)
        # ディレクトリwo移動
        shutil.copyfile(out_path, output_dir_img+ '/' + framename2+ '.jpg')
        #shutil.move(out_path, output_dir_img)
