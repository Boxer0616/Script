import glob
import random
import os
import xml.etree.ElementTree as ET
from PIL import Image
import copy
import cv2
import argparse
import pyclipper
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category',    default = './category.txt')
parser.add_argument('-i', '--input_dir', default = './dataset/20191011_形状最終/test')
parser.add_argument('-o', '--output_dir', default = './output')
parser.add_argument('-s', '--size', default = '800')
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
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

def clip_image(img, x, y, w, h):
    #box=(left, upper, right, lower)
    im_crop = img.crop((x, y, x+w, y+h))
    return im_crop

def clip_fix_data(img, x, y, xml):

    clipped_img = clip_image(img, x, y, int(args.size), int(args.size))
    clipped_bboxes = clip_bbox(xml, x, y, int(args.size), int(args.size))


    return clipped_img, clipped_bboxes


def clip_bbox(anno, x_from, y_from, w, h):

    ann = copy.deepcopy(anno)
    x_to = x_from+w
    y_to = y_from+h

    objs = ann.findall('object')

    #object
    for obj in objs:
        bndbox_anno = obj.find('bndbox')
        xmin = int(int(bndbox_anno.find('xmin').text))
        xmax = int(int(bndbox_anno.find('xmax').text))
        ymin = int(int(bndbox_anno.find('ymin').text))
        ymax = int(int(bndbox_anno.find('ymax').text))

        #頂点数2以下ポリゴン削除
        polygon_num = int(obj.find('./polygon/num').text)
        if polygon_num <= 2:
            print("invalid")
            ann.getroot().remove(obj)
            continue
            


        #対象矩形範囲外の場合
        if x_to < xmin or xmax < x_from or \
            y_to < ymin or ymax < y_from:
            #print("continue", x_from, y_from, x_to, y_to)
            ann.getroot().remove(obj)
            continue
        '''
        if False:
           # 一部が矩形をまたがっている場合と内包している場合
            xmin = max([xmin, x_from])
            ymin = max([ymin, y_from])
            xmax = min([xmax, x_to])
            ymax = min([ymax, y_to])
        '''

        if int(xmin-x_from)<=0:
            bndbox_anno.find('xmin').text = '0'
        else:
            bndbox_anno.find('xmin').text = str(xmin-x_from)
        if int(xmax-x_from) >= w:
            bndbox_anno.find('xmax').text = str(w)
        else:
            bndbox_anno.find('xmax').text = str(xmax-x_from)
        if int(ymin-y_from) <=0:
            bndbox_anno.find('ymin').text = '0'
        else:
            bndbox_anno.find('ymin').text = str(ymin-y_from)
        if int(ymax-y_from) >= h:
            bndbox_anno.find('ymax').text = str(h)
        else:
            bndbox_anno.find('ymax').text = str(ymax-y_from)

    for x in ann.findall('./object/polygon/point/x'):
        x.text = str(int(int(x.text)-x_from))

    for y in ann.findall('./object/polygon/point/y'):
        y.text = str(int(int(y.text)-y_from))

    objs2 = ann.findall('object')

    #クリップ領域
    clip = ((0,800),(800,800),(800,0),(0,0))
    for objj in objs2:
        polys = []
        polygons = []
        clip_polygon = []
        polygon_dert = objj.findall('.//polygon')
        points = objj.findall('.//polygon/point')
        for p in points:
            x = int(p.find('./x').text)
            y = int(p.find('./y').text)
            polys.append((x,y))

        polygons.append(polys)
        #print(polygons)
        area = cv2.contourArea(np.array(polygons))
        #print(area)
        if area > 0:
            #ポリゴンをクリップ
            clip_polygon = clip_poly(clip,polygons)
        else:#面積が0だったらオブジェクトを削除
            objs2.remove(objj)

        #ポリゴンがクリップされたら変更
        if len(clip_polygon) != 0:
            for dert in polygon_dert:
                objj.remove(dert)

            new_poly = ET.SubElement(objj,'polygon')
            for n_po in clip_polygon:
                for n_poly in n_po:
                    new_point = ET.SubElement(new_poly,'point')
                    new_x = ET.SubElement(new_point,'x')
                    new_x.text = str(n_poly[0])
                    new_y = ET.SubElement(new_point,'y')
                    new_y.text = str(n_poly[1])

    return ann

#ポリゴンクリップ
def clip_poly(clip,polygons):

    pc = pyclipper.Pyclipper()
    pc.AddPath(clip, pyclipper.PT_CLIP, True)
    pc.AddPaths(polygons, pyclipper.PT_SUBJECT, True)
    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

    return solution

'''
    pc = pyclipper.Pyclipper()
    #pco = pyclipper.PyclipperOffset()
    subj = (
    ((180, 200), (260, 200), (260, 150), (180, 150)),
    ((215, 160), (230, 190), (200, 190))
    )
    clip = ((190, 210), (240, 210), (240, 130), (190, 130))
    rr = ((0,800),(800,800),(800,0),(0,0))
    retu = []
    #for polygo in polygons:
     #   po = []
        #pc.AddPath(rr, pyclipper.PT_CLIP, True)
        #pc.AddPaths(polygo, pyclipper.PT_SUBJECT, True)
    #po.append(polygo)
    pc.AddPath(rr, pyclipper.PT_CLIP, True)
    pc.AddPaths(polygons, pyclipper.PT_SUBJECT, True)
    #pco.AddPaths(polygo, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    #pco.Execute(-7.0)
    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    #retu.append(solution)
'''


# 指定数分割
def devide():
    #パスの読み込み
    img_path_list,xml_path_list = import_data()
    for i,ann in enumerate(xml_path_list):
        dir_name,file_name = path_to_name(ann)
        img = img_path_list[i]
        f = Image.open(img)
        shape = f.size
        count = 0
        #f = cv2.imread(img)
        xml = ET.parse(ann)
        for start_h in range(0, shape[0], 256):
            count2=0
            for start_w in range(0, shape[1], 256):
                start_h_new = start_h
                start_w_new = start_w
                if start_h + int(args.size) > shape[0]:
                    start_h_new = start_h - ((start_h + int(args.size)) - shape[1])
                if start_w + int(args.size) > shape[1]:
                    start_w_new = start_w - ((start_w + int(args.size)) - shape[1])
                if (start_w_new+int(args.size)) >= 2560:
                    count2 += 1
                if count2 <= 1:
                    if not (os.path.isdir(os.path.join(args.output_dir, 'image'))):
                        os.makedirs(os.path.join(args.output_dir, 'image'))
                    if not (os.path.isdir(os.path.join(args.output_dir, 'label'))):
                        os.makedirs(os.path.join(args.output_dir, 'label'))
                    cimg, cxml = clip_fix_data(f, start_w_new, start_h_new, xml)
                    cxml = clip_bbox(xml, start_w_new, start_h_new, int(args.size), int(args.size))
                    #img保存
                    save_img_path = os.path.join(args.output_dir,'image', "%s_%04d_%04d.jpg" % (file_name, start_h_new, start_w_new))
                    cimg.save(save_img_path)
                    #xml保存
                    save_xml_path = os.path.join(args.output_dir,'label', "%s_%04d_%04d.xml" % (file_name, start_h_new, start_w_new))
                    cxml.write(save_xml_path)
                    count += 1

if __name__ == '__main__':

    # 分割
    devide()
