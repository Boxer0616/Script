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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category',      default = './category2.txt')
parser.add_argument('-i', '--input_dir',  default = './dataset/predict/xml')
parser.add_argument('-ii', '--input_dir2',  default = './dataset/ketugoukekka_nms/ketubgou_nms/xml')
parser.add_argument('-o', '--out_dir',  default = './output')
args = parser.parse_args()

#パスの読み込み
def import_data():
    xml_path_list = glob.glob(os.path.join(args.input_dir,'**' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)
    xml_path_list2 = glob.glob(os.path.join(args.input_dir2,'**' ,'*.xml'))
    xml_path_list2 = sorted(xml_path_list2)

    return xml_path_list,xml_path_list2

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#矩形を取得
def get_bbox_polygon(xml_path_list):
    #オブジェクトをxmlごとに格納
    obj_dict = []
    #polygons = [] 
    obj_cnt = []
    cnt = 0
    for xml_path in xml_path_list:
        #オブジェクトリスト初期化
        objects = []
        #xml解析
        tree = etree.parse(xml_path)
        filename = tree.find('filename').text
        obj_cnt.append(cnt)
        cnt = 0

        #object検索
        objs = tree.xpath('//object')
        for obj in objs:
            #polys = []
            #矩形を取得
            bdbs = obj.xpath('bndbox')
            #category取得
            cate = obj.find('category') 
            for bdb in bdbs:
                objects.append({
                'xmin' : int(bdb.xpath('.//xmin')[0].text),
                'xmax' : int(bdb.xpath('.//xmax')[0].text),
                'ymin' : int(bdb.xpath('.//ymin')[0].text),
                'ymax' : int(bdb.xpath('.//ymax')[0].text),
                'category' : cate.find('value').text
                })
            #ポリゴン取得
            '''
            points = obj.xpath('.//polygon/point')
            for p in points:
                x = int(p.xpath('.//x')[0].text)
                y = int(p.xpath('.//y')[0].text)
                polys.append([x,y])
            polygons.append(polys)
            '''
        #obj_dict[filename] = objects
        obj_dict.append(objects)
    del obj_cnt[0]
    return obj_dict

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
        nn = []
        obj_count = 0
        #オブジェクトごと
        for obj in objs:
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
                                'filename':filename,
                                'boxlabel':str(obj_count) + '_' +filename})
            else:
                for bdb in bdbs:
                    objj.append({'xmin' :int(bdb.xpath('.//xmin')[0].text),
                                'ymin' :int(bdb.xpath('.//ymin')[0].text),
                                'xmax' :int(bdb.xpath('.//xmax')[0].text),
                                'ymax' :int(bdb.xpath('.//ymax')[0].text),
                                'category' :cc,
                                'filename':filename,
                                'boxlabel':str(obj_count) + '_' +filename})
            obj_count += 1

        arr_obj = np.array(objj)
        obj_m.append(arr_obj)
    return obj_m

#xmlファイル作成
def create_xml(obj):
    if len(obj)!=0:
        nn = obj[0]['filename'].rsplit("_")
        file = nn[0] + '_' + nn[1] + '_' + nn[2] + '_' + nn[3]	
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
        directory = nn[0] + '_' + nn[1]
        folder.text = directory
        filename.text = obj[0]['filename']

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
            #scrr = ET.SubElement(objc, 'score')
            #scrr.text = str(scores[i])
            xmin= ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax= ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')

            xmin.text = str(dd['xmin'])
            ymin.text = str(dd['ymin'])
            xmax.text = str(dd['xmax'])
            ymax.text = str(dd['ymax'])

            # category
            category = ET.SubElement(objc, 'category')
            value = ET.SubElement(category, 'value')
            value.text = str(dd['category'])

        #tree = ET.ElementTree(root)
        #tree.write(xml_path)
        # ツリー反映
        tree = ET.ElementTree(root)
        #改行
        indent(root)
        create_dir(directory)
        #xml保存
        tree.write(os.path.join(args.out_dir, directory) + '/' + file  + '.xml', encoding="utf-8", xml_declaration=True)

#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory))):
        os.makedirs(os.path.join(args.out_dir, directory))

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

#IoUの計算
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA['xmin'], boxB['xmin'])
    yA = max(boxA['ymin'], boxB['ymin'])
    xB = min(boxA['xmax'], boxB['xmax'])
    yB = min(boxA['ymax'], boxB['ymax'])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #interArea = iw * ih
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
    boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

#検出未検出はんてい
def detection_or_not_detection(boA, boB):
    # determine the (x, y)-coordinates of the intersection rectangle
    dete = []
    detection_box = []
    notdetection_box = []
    #正解boxに推論box総当りでIou計算
    for a,boxA in enumerate(boA):
        dete = []
        not_dete = []
        for b,boxB in enumerate(boB):
            #Iou計算
            iou = bb_intersection_over_union(boxA,boxB)
            #Iouがしきい値以上
            if iou >= 0.3:
                dete.append({'iou':iou,
                             'boxA':boxA,
                             'boxB':boxB})
        #Iouが重なっている正解ボックスと推論ボックスを格納
        detection_box.append(dete)
        if len(detection_box[a]) == 0:
            #一つも重なる推論ボックスがなかった正解ボックスを格納
            notdetection_box.append({'iou':iou,
                                     'boxA':boxA,})
    #空白を削除
    dd = [i for i in detection_box if len(i)!=0]
		
    return dd,notdetection_box

##ダブリと不要
def classify_iou(iou_list):
    # ダブっている情報を削除
    for i, x in enumerate(iou_list[:]):
        for j, y in enumerate(iou_list[:]):
            if (x[0]['boxA']['filename'] == y[0]['boxA']['filename'] and x[0]['boxA']['boxlabel'] != y[0]['boxA']['boxlabel'] and x[0]['boxB']['boxlabel'] == y[0]['boxB']['boxlabel']):
                # iouが小さい方を削除
                if (x[0]['iou'] >= y[0]['iou']):
                    iou_list.remove(y)

                elif (x[0]['iou'] ==y[0]['iou']):
                    if (x[0]['boxB']['score'] >= y[0]['boxB']['score'] and y[0]['boxA']['category'] != y[0]['boxB']['category']):
                        iou_list.remove(y)

                    elif (x[0]['boxA']['category'] == x[0]['boxB']['category']):
                        iou_list.remove(y)

    # 不要
    unnecessary = []

    # リストから"不要"を取り出す
    for i, x in enumerate(iou_list[:]):
        for j, y in enumerate(iou_list[:]):
            if (x[0]['boxA']['filename']  == y[0]['boxA']['filename'] and x[0]['boxA']['boxlabel'] == y[0]['boxA']['boxlabel'] and x[0]['boxB']['boxlabel'] != y[0]['boxB']['boxlabel']):
                # iouが小さい方を"不要"に
                if (x[0]['iou'] >= y[0]['iou']):
                    unnecessary.append(y)
                    iou_list.remove(y)

                elif (x[0]['iou'] == y[0]['iou']):
                    if (x[0]['boxB']['score'] >= y[0]['boxB']['score'] and y[0]['boxA']['category'] != y[0]['boxB']['category']):
                        #print (x, y)
                        unnecessary.append(y)
                        iou_list.remove(y)

                    elif (x[0]['boxA']['category'] == x[0]['boxB']['category']):
                        #print (x, y)
                        unnecessary.append(y)
                        iou_list.remove(y)

    # 検出（正解＋推論）
    detection = iou_list

    return detection, unnecessary

##評価数ごとにカウント
def count_evaluation_num(categories,pre_boxnum_list,cor_boxnum_list,detection, unnecessary,ob1):
    # 静的確保
    count_eval = [[0 for i in range(len(categories)*2 + 3)] for j in range(len(categories))]

    bbox = []
    # 検出（正解＋推論）
    for i, x in enumerate(categories):
        for j, y in enumerate(categories):
            for c,obj in enumerate(detection):
                if (x == obj[0]['boxA']['category'] and y == obj[0]['boxB']['category']):
                    count_eval[i][1 + j] += 1
                if (x == obj[0]['boxB']['category'] and y == obj[0]['boxA']['category']):
                    count_eval[i][1 + len(categories) + j] += 1
                   
    # 不要
    for i, category in enumerate(categories):
        for obj in unnecessary:
            if (category == obj[0]['boxB']['category']):
                count_eval[i][2 + len(categories)*2] += 1
                bbox.append(c)

    # 過検出
    for i, box in enumerate(count_eval):
        pre_box_num  = 0

        for j in range(1 + len(categories), len(categories)*2 + 3):
            pre_box_num += count_eval[i][j]

        t = pre_boxnum_list[i][1] - pre_box_num
        count_eval[i][len(categories)*2 + 1] = t

    # 未検出
    for i, box in enumerate(count_eval):
        cor_box_num  = 0

        for j in range(1, len(categories) + 1):
            cor_box_num += count_eval[i][j]

        t = cor_boxnum_list[i][1] - cor_box_num
        count_eval[i][0] = t

    # count_eval = [未検出, 正解検出(1), ・・・, 正解検出(n), 推論検出(1), ・・・, 推論検出(n), 過検出, 不要]
    return count_eval

#未検出矩形を取得
def get_miss_box(detection,ob1):
    ob1_list = ob1.tolist()
    for c,obbe in enumerate(ob1_list[:]):
        for obj in detection:
            if obbe['boxlabel'] == obj[0]['boxA']['boxlabel']:
                ob1_list.remove(obbe)
    
    return ob1_list


"""
---------------------------------------------------------------------------------------------------

ボックス合計数を取得

---------------------------------------------------------------------------------------------------
"""
def get_boxnum(label_list, CLASSES):
    boxnum_list = []

    for category in CLASSES:
        cnt = [x['category'] for x in label_list].count(category)

        boxnum_list.append([category, cnt])

    return boxnum_list

#未検出と検出ボックス取得
def get_dete_notdete_box(categories,obje1,obje2):
    detection_box_list = []
    not_detection_box_list = []
    detection_box_list2 = []
    add_detection_box_list = []
    count_miss_fi = 0
    count_miss_gfi = 0
    count_miss_hfi = 0
    for p,(ob1,ob2) in enumerate(zip(obje1,obje2)):
        #ボックスの数
        corbox_num_list = get_boxnum(ob1,categories)
        prebox_num_list = get_boxnum(ob2,categories)
        #しきい値以上のIouをもつボックスの組み合わせを返す
        detection_box,not_detection_box = detection_or_not_detection(ob1, ob2)
        #ダブリを削除、不要を抽出
        detection,unnecessary = classify_iou(detection_box)
        #未検出矩形を取り出す
        rr = count_evaluation_num(categories,prebox_num_list,corbox_num_list,detection, unnecessary,ob1)
        miss_box = get_miss_box(detection,ob1)
        count_miss_fi = rr[0][0]+count_miss_fi
        count_miss_gfi = rr[1][0]+count_miss_gfi
        count_miss_hfi = rr[2][0]+count_miss_hfi
        not_detection_box_list.append(miss_box)
        '''
        #推論ボックスに対する
        detection_box2,add_detection_box = detection_or_not_detection(ob2, ob1)
        
        #正解ボックスに対する検出と未検出
        detection_box_list.append(detection_box)
        not_detection_box_list.append(not_detection_box)

        #推論ボックスに対する検出と過検出
        detection_box_list2.append(detection_box2)
        add_detection_box_list.append(add_detection_box)
        '''

    return not_detection_box_list

#検出ボックス判定(未検出を返す)
def exact_detectionbox(detection_box_list):
    save = []
    for detection_box in detection_box_list:
        min_bbb = []
        for detection in detection_box:
            if len(detection) > 1:
                io = []
                for detect in detection:
                    io.append(detect['iou'])
                max_index = io.index(max(io))
                min_index = io.index(min(io))
                max_b = detection[max_index]
                #推論ボックスにかぶった正解ボックスのうち一番Iouが低いものを取得
                min_b = detection[min_index]
                min_bbb.append({'boxA':min_b['boxB']})
        save.append(min_bbb)

    return save

def add_list(notdetection_box_list,save):
    add = []
    for s,notdetection_box in enumerate(notdetection_box_list):
        aaa = notdetection_box+save[s]
        add.append(aaa)
    
    return add

def caburi_remove(detection_box_list,save):
    for mike in save:
        if len(mike) !=0:
            for nn,mi in enumerate(mike):
                for ken in detection_box_list:
                    for kk in ken:
                        count = 0
                        for k in kk:
                            if k['boxA']['boxlabel'] == mi['boxA']['boxlabel'] and count==0:
                                mike.remove(mi)
                                count +=1

# カテゴリの読み込み
def read_category():
    with open(args.category, 'r') as f:
        categories = f.read().splitlines()
    
    return categories

if __name__ == '__main__':
    #ファイルパスの取得
    xml_path_list,xml_path_list2 = import_data()
    #カテゴリー
    categories = read_category()
    #正解矩形の取得
    #obje1 = get_bbox_polygon(xml_path_list)
    obje1 = get_bbox(xml_path_list)
    #推論結果矩形の取得
    #obje2 = get_bbox_polygon(xml_path_list2)
    obje2 = get_bbox(xml_path_list2)
    #未検出(一つも推論ボックスとかぶらなかった正解ボックス)と検出(推論ボックスからみた正解ボックス)の取得
    notdetection_box_list = get_dete_notdete_box(categories,obje1,obje2)


    #xmlに書き込み
    for pp,obb in enumerate(notdetection_box_list):
        #xml作成
    	create_xml(obb)
'''
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
'''
