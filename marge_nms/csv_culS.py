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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './output-nms/presfam-800marge/draw-nms-0.6')
parser.add_argument('-o', '--out_dir',  default = './output-nms/presfam-800marge')
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

#csv
def write_csv(rotate):
    save_row = {}

    with open(os.path.join(args.input_dir,'some.csv'),'w') as f:
        writer = csv.DictWriter(f, fieldnames=rotate[0].keys(),delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(rotate[0].keys())
        length = len(rotate)

        for i in range(length):
            for k, vs in rotate[i].items():
                save_row[k] = vs

            writer.writerow(save_row)

#xmlから矩形を取得
def get_bbox(xml_path):         
    obj_m = []
    polygon_m = []
    for n, xml in enumerate(xml_path):
        tree = etree.parse(xml)
        filename = tree.find('filename').text
        objs = tree.xpath('//object')
        objj = []
        obj_count = 0
        poly=[]
        #オブジェクトごと
        for obj in objs:
            po={}
            bdbs = obj.xpath('bndbox') 
            #スコアの取得
            scoree = obj.xpath('score')
            for scor in scoree:
                sc = float(scor.text)
            #カテゴリの取得
            cate = obj.xpath('.//category/value')
            for scr in cate:
                cc = scr.text
            #bdbを取得
            if len(scoree) !=0:
                for bdb in bdbs:
                    objj.append({'xmin' :int(bdb.xpath('.//xmin')[0].text),
                                'ymin' :int(bdb.xpath('.//ymin')[0].text),
                                'xmax' :int(bdb.xpath('.//xmax')[0].text),
                                'ymax' :int(bdb.xpath('.//ymax')[0].text),
                                'category' :cc,
                                'score' :sc,
                                'filename':filename,})
            else:
                for bdb in bdbs:
                    objj.append({'xmin' :int(bdb.xpath('.//xmin')[0].text),
                                'ymin' :int(bdb.xpath('.//ymin')[0].text),
                                'xmax' :int(bdb.xpath('.//xmax')[0].text),
                                'ymax' :int(bdb.xpath('.//ymax')[0].text),
                                'category' :cc,
                                'filename':filename,})

            obj_count += 1

            #ポリゴンの取得
            if obj.xpath('rbndbox'):
                points = obj.xpath('rbndbox')
                for m,p in enumerate(points):
                    for g in range(4):
                        po['x'+str(g)] = int(p.xpath('.//x' + str(g))[0].text)
                        po['y'+str(g)] = int(p.xpath('.//y' + str(g))[0].text)
                po['category'] = cc
                po['filename'] = filename
                po['boxlabel'] = str(obj_count) + '_' +filename
                po['angle'] = int(bdb.xpath('.//angle')[0].text)
                po['area'] = int(bdb.xpath('.//polyarea')[0].text)
                if po==[]:
                    x = 0
                    y = 0
                    po['x'+str(0)] = x
                    po['y'+str(0)] = y
                    #poly.append([int(p.xpath('.//x')[0].text),int(p.xpath('.//y')[0].text)])
                
                poly.append(po)
        #array型にする     
        arr_poly = np.array(poly)
        polygon_m.append(arr_poly)
        arr_obj = np.array(objj)
        obj_m.append(arr_obj)
    return obj_m,polygon_m

#xmlに書き込みと保存
def write_xml(xml_path_list,b,):
    count = 0
    for xc,xml_path in enumerate(xml_path_list):
        x = [''] * 4
        y = [''] * 4
        tree = ET.parse(xml_path)
        # ツリーを取得
        root = tree.getroot()
        filename = tree.find('filename').text
        directory = tree.find('folder').text
        for bc,obje in enumerate(tree.findall('object')): 
            #xminとか削除
            #xmin_etc_remove(bndbox,'xmin')
            #xmin_etc_remove(bndbox,'ymin')
            #xmin_etc_remove(bndbox,'xmax')
            #xmin_etc_remove(bndbox,'ymax')
            #書き込みノード追加
            rea = ET.SubElement(obje,'polyArea')
            r_bdb = tree.find('rbndbox')
            angle = obje.find('bndbox/angle')
            bdb_area = obje.find('bndbox/rbndarea')
            #obje.insert(4, obje[-1])        
            #回転矩形の座標書き込み
            '''
            for n in range(4):
                x[n] = ET.SubElement(r_bdb, 'x'+str(n))
                y[n] = ET.SubElement(r_bdb, 'y'+str(n))

            for i in range(4):
                if b[xc][bc]['x'+str(i)] < 0:
                    x[i].text = '0'
                else:
                    x[i].text = str(b[xc][bc]['x'+str(i)])
                if b[xc][bc]['y'+str(i)] < 0:
                    y[i].text = '0'
                else:
                    y[i].text = str(b[xc][bc]['y'+str(i)])
            '''
            #ポリゴンの面積
            rea.text = str(b[xc][bc]['area'])
            #回転矩形の角度
            angle.text = str(b[xc][bc]['angle'])
            #回転矩形の面積
            bdb_area.text = str(b[xc][bc]['men'])
            count += 1

        # ツリー反映
        tree = ET.ElementTree(root)
        #改行
        indent(root)
        #ディレクトリ作成
        create_dir(directory+'/supervised')
        #xml保存
        tree.write(os.path.join(args.out_dir, directory)+ '/' +'supervised'+ '/' +filename+'.xml', encoding="utf-8", xml_declaration=True)

#xmin,ymin,xma,ymax削除
def xmin_etc_remove(bndbox,remove_name):
    for e,re_neme in enumerate(bndbox.findall(remove_name)):

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
'''
#保存パス
def get_save_path(save_dir,file_idx,bdb):
    
    path = os.path.jo    polys.append(([polyss['xmin'],polyss['ymin']]))
    polys.append(([polyss['xmin'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymin']]))ave_dir,"%s_%04d_%04d_%04d_%04d.jpg" % (file_idx, bdb['xmin'], bdb['ymin'],bdb['xmax'], bdb['ymax']))

    return path
'''
#回転矩形かあ
def get_rotate_bdb(xm    polys.append(([polyss['xmin'],polyss['ymin']]))
    polys.append(([polyss['xmin'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymin']]))th_list,poly):
    #rotate_bbox = []    polys.append(([polyss['xmin'],polyss['ymin']]))
    polys.append(([polyss['xmin'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymin']]))
    for s,po in enumerate(poly):
        #rotate_list = []
        for i, cnt in enumerate(po):
            rotate_dic={}
            #ライブラリで計算できる型に変更
            cnn = get_polyxy(cnt)
            cnn = np.array(cnn)
            # 回転した長方形の4点の座標を取得する。
            #rect_points = cv2.boxPoints(cnn)
            #ポリゴンの面積
            men = cv2.contourArea(cnn)
            cnt['men']=men
            # 輪郭に外接する回転した長方形を取得する。
            #rect = cv2.minAreaRect(cnn)
            #(cx, cy), (width, height), angle = rect
            #print('bounding box of contour {} => '
            #    'center: ({:.2f}, {:.2f}), size: ({:.2f}, {:.2f}), angle: {:.2f}'.format(
            #    i, cx, cy, width, height, angle))
            # 回転した長方形の4点の座標を取得する。
            #rect_points = cv2.boxPoints(rect)
            #回転矩形の面積
            #kukei_men = cv2.contourArea(cnn)
            #adf = width*height
            #rect_points = np.array(rect_points,dtype=np.int64)
            #辞書リスト型に
            #change_polu(cnt,rotate_dic,rect_points,angle,men,kukei_men,rotate_list)
        #rotate_bbox.append(rotate_list)
        
    return poly

#ポリゴンの座標取得
def get_polyxy(polyss):
    polys = []
    l = len(polyss)-3
    polys.append(([polyss['xmin'],polyss['ymin']]))
    polys.append(([polyss['xmin'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymin']]))
    return polys

#辞書リスト型に
def change_polu(cnt,rotate_dic,rect_points,angle,men,kukei_men,rotate_list):
    rect = rect_points.tolist()
    for i,r in enumerate(rect):
        rotate_dic['x'+str(i)] = r[0]
        rotate_dic['y'+str(i)] = r[1]
    rotate_dic['category'] = cnt['category']
    rotate_dic['filename'] = cnt['filename']
    rotate_dic['boxlabel'] = cnt['boxlabel']
    if int(angle*-1)>45:
        rotate_dic['angle'] = 90-int(angle*-1)
    else:
        rotate_dic['angle'] = int(angle*-1)
    rotate_dic['area'] = men
    rotate_dic['bdbarea'] = kukei_men

    rotate_list.append(rotate_dic)

#hitomatome
def one_list(rotate_bdb):
    one = []
    for rotate in rotate_bdb:
        for rota in rotate:
            one.append(rota)
    return one   



#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory))):
        os.makedirs(os.path.join(args.out_dir, directory))

if __name__ == '__main__':
    #データのパス
    img_path_list,xml_path_list = import_data()
    #矩形とポリゴンを取得
    bdb,polygonn = get_bbox(xml_path_list)
    #bdbで切り出し
    #for image in img_path_list:
    #    dire,name = path_to_name(image)
    #    create_dir(dire)
    #    save_dir = os.path.join(args.out_dir,dire)
    #    for bd in bdb[name]:
    #        img_name = divide_bdb_image(name, image, bd, save_dir)
    #回転矩形を取得
    rotate_bdb = get_rotate_bdb(xml_path_list,bdb)
    #面積
    v = one_list(rotate_bdb)
    write_csv(v)
    #xmlに書き込み
    #write_xml(xml_path_list,rotate_bdb)
