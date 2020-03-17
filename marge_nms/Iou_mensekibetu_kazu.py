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
import pdb;
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category',      default = './category.txt')
parser.add_argument('-i', '--input_dir',  default = '/home/hiroshi/Dataset/folda-change/20191011矩形最終/remove-edge-val')
parser.add_argument('-ii', '--input_dir2',  default = './output-nms/presfam-800marge/draw-nms-0.6')
parser.add_argument('-ou', '--iou_threshold', help = 'Threshold of IoU', type = float, default = 0.1)
parser.add_argument('-s', '--menseki_threshold', help = 'Threshold of IoU', type = int,default = 250)
parser.add_argument('-o', '--out_dir',  default = './output')
args = parser.parse_args()

#パスの読み込み
def import_data():
    xml_path_list = glob.glob(os.path.join(args.input_dir,'**' , 'supervised' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)
    xml_path_list2 = glob.glob(os.path.join(args.input_dir2,'**' , 'supervised' ,'*.xml'))
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

#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory,'supervised'))):
        os.makedirs(os.path.join(args.out_dir, directory,'supervised'))


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

def calculate_iou(correct_label_list,predict_label_list):
    iou_list = []

    for boxA in correct_label_list:
        for boxB in predict_label_list:
            if  boxB['category'] != 'space' and boxA['category'] != 'space':
                if boxA['filename']==boxB['filename']:
                    xA = max(boxA['xmin'], boxB['xmin'])
                    yA = max(boxA['ymin'], boxB['ymin'])
                    xB = min(boxA['xmax'], boxB['xmax'])
                    yB = min(boxA['ymax'], boxB['ymax'])
                    K_area = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)
                    N_area = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)

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

                    if (args.iou_threshold <= iou):
                        iou_list.append({'iou':iou,'boxA':boxA, 'boxB':boxB})

    # 昇順にソート
    #iou_list.sort()

    # iou_list = [inference/directory/framename, cor_label, pre_label, cor_num, pre_num, iou, score]
    return iou_list

#検出未検出はんてい
def detection_or_not_detection2(boA, boB):
    # determine the (x, y)-coordinates of the intersection rectangle
    dete = []
    detection_box = []
    notdetection_box = []
    dd = []
    #正解boxに推論box総当りでIou計算
    for a,boxA in enumerate(boA):
        dete = []
        not_dete = []
        for b,boxB in enumerate(boB):
            #Iou計算
            iou = bb_intersection_over_union(boxA,boxB[0]['boxB'])
            #Iouがしきい値以上
            if iou >= 0.1:
                dete.append({'iou':iou,
                             'boxA':boxA,
                             'boxB':boxB[0]['boxB']})
        #Iouが重なっている正解ボックスと推論ボックスを格納
        detection_box.append(dete)
        if len(detection_box[a]) == 0:
            #一つも重なる推論ボックスがなかった正解ボックスを格納
            notdetection_box.append({'iou':iou,
                                     'boxA':boxA,})
    #空白を削除
    dd = [i for i in detection_box if len(i)!=0]
		
    return dd,notdetection_box

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
            boxA_S = get_bnd_S(boxA)
            boxB_S = get_bnd_S(boxB)
            #Iouがしきい値以上
            if iou >= 0.1:
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
    for i, xx in enumerate(iou_list[:]):
        for f, x in enumerate(xx):
            for p, yy in enumerate(iou_list[:]):
                for j, y in enumerate(yy):
                    if (x['boxA']['boxlabel'] != y['boxA']['boxlabel'] and x['boxB']['boxlabel'] == y['boxB']['boxlabel']):
                        # iouが小さい方を削除
                        if (x['iou'] >= y['iou']):
                            yy.remove(y)

                        elif (x['iou'] ==y['iou']):
                            if (x['boxB']['score'] >= y['boxB']['score'] and y['boxA']['category'] != y['boxB']['category']):
                                yy.remove(yy)

                            elif (x['boxA']['category'] == x['boxB']['category']):
                                yy.remove(y)

    # 不要
    unnecessary = []

    # リストから"不要"を取り出す
    for i, xx in enumerate(iou_list[:]):
        for i, x in enumerate(xx):
            for i, yy in enumerate(iou_list[:]):
                for j, y in enumerate(yy):
                    if (x['boxA']['boxlabel'] == y['boxA']['boxlabel'] and x['boxB']['boxlabel'] != y['boxB']['boxlabel']):
                        # iouが小さい方を"不要"に
                        if (x['iou'] >= y['iou']):
                            unnecessary.append(yy)
                            yy.remove(y)

                        elif (x['iou'] == y['iou']):
                            if (x['boxB']['score'] >= y['boxB']['score'] and y['boxA']['category'] != y['boxB']['category']):
                                #print (x, y)
                                unnecessary.append(yy)
                                yy.remove(y)

                            elif (x['boxA']['category'] == x['boxB']['category']):
                                #print (x, y)
                                unnecessary.append(yy)
                                yy.remove(y)
    # 検出（正解＋推論）
    detection = iou_list
    #空白を削除
    dd = [i for i in detection if len(i)!=0]
    return dd, unnecessary

##ダブリと不要
def classify_iou_2(iou_list_t):
    # ダブっている情報を削除
    for d, a in enumerate(iou_list_t):
            for p, b in enumerate(iou_list_t):
                    if (a['boxA']['boxlabel'] != b['boxA']['boxlabel'] and a['boxB']['boxlabel'] == b['boxB']['boxlabel']):
                        # iouが小さい方を削除
                        if (a['iou'] >= b['iou']):
                            last_num = iou_list_t.pop(p)
                            iou_list_t.append(last_num)
                            iou_list_t.pop()

                        elif (a['iou'] ==b['iou']):
                            if (a['boxB']['score'] >= b['boxB']['score'] and b['boxA']['category'] != b['boxB']['category']):
                                last_num = iou_list_t.pop(p)
                                iou_list_t.append(last_num)
                                iou_list_t.pop()

                            elif (a['boxA']['category'] == a['boxB']['category']):
                                last_num = iou_list_t.pop(p)
                                iou_list_t.append(last_num)
                                iou_list_t.pop()

    # 不要
    unnecessary = []

    # リストから"不要"を取り出す
    for i, x in enumerate(iou_list_t[:]):
            for j, y in enumerate(iou_list_t[:]):
                    if (x['boxA']['boxlabel'] == y['boxA']['boxlabel'] and x['boxB']['boxlabel'] != y['boxB']['boxlabel']):
                        # iouが小さい方を"不要"に
                        if (x['iou'] >= y['iou']):
                            unnecessary.append(y)
                            last_num = iou_list_t.pop(j)
                            iou_list_t.append(last_num)
                            iou_list_t.pop()

                        elif (x['iou'] == y['iou']):
                            if (x['boxB']['score'] >= y['boxB']['score'] and y['boxA']['category'] != y['boxB']['category']):
                                #print (x, y)
                                unnecessary.append(y)
                                last_num = iou_list_t.pop(j)
                                iou_list_t.append(last_num)
                                iou_list_t.pop()

                            elif (x['boxA']['category'] == x['boxB']['category']):
                                #print (x, y)
                                unnecessary.append(y)
                                last_num = iou_list_t.pop(j)
                                iou_list_t.append(last_num)
                                iou_list_t.pop()
    # 検出（正解＋推論）
    detection = iou_list_t
    #空白を削除
    dd = [i for i in detection if len(i)!=0]
    return dd, unnecessary



##評価数ごとにカウント
def count_evaluation_num(categories,pre_boxnum_list,cor_boxnum_list,detection, unnecessary):
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
        if t <= 0:
            count_eval[i][len(categories)*2 + 1] = 0
        else:
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
    obb = np.array(ob1)
    ob1_list = obb.tolist()
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
    evaluation_num = []
    unnecessary_list = []
    not_detection_box_list = []
    detection_box_list2 = []
    maxAA = []
    maxBB = []
    mensekilistA = []
    mensekilistB = []
    add_detection_box_list = []
    count_miss_fi = 0
    count_miss_gfi = 0
    count_miss_hfi = 0
    miss_box = []
    for p,ob1 in enumerate(obje1):
        for ob2 in obje2:
            #print('ob1_len=',len(ob1))
            #print('ob2_len=',len(ob2))
            if ob1[0]['filename']==ob2[0]['filename']:
                #ボックスの数
                corbox_num_list = get_boxnum(ob1,categories)
                prebox_num_list = get_boxnum(ob2,categories)
                #しきい値以上のIouをもつボックスの組み合わせを返す
                detection_box,not_detection_box = detection_or_not_detection(ob1, ob2)
                maxA,menseki_listA = get_max_S(ob1)
                maxB,menseki_listB = get_max_S(ob2)
                #ダブリを削除、不要を抽出
                detection,unnecessary = classify_iou(detection_box)
                #未検出矩形を取り出す
                #rr = count_evaluation_num(categories,prebox_num_list,corbox_num_list,detection, unnecessary)
                miss_box = get_miss_box(detection,ob1)
                not_detection_box_list.append(miss_box)
                detection_box_list2.append(detection)
                #evaluation_num.append(rr)
                unnecessary_list.append(unnecessary)
                maxAA.append(maxA)
                mensekilistA.append(menseki_listA)
                mensekilistB.append(menseki_listB)
                maxBB.append(maxB)
    max_boxA = max(maxAA)
    max_boxB = max(maxBB)

    return not_detection_box_list,detection_box_list2,unnecessary_list,max_boxA,max_boxB

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

#矩形の面積
def get_bnd_S(poly):

    rotate_dic={}
    #ライブラリで計算できる型に変更
    cnn = get_polyxy(poly)
    cnn = np.array(cnn)
    men = cv2.contourArea(cnn)
    poly['menseki'] = men
        
    return men

#矩形の座標取得
def get_polyxy(polyss):
    polys = []
    l = len(polyss)-3
    polys.append(([polyss['xmin'],polyss['ymin']]))
    polys.append(([polyss['xmin'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymax']]))
    polys.append(([polyss['xmax'],polyss['ymin']]))
    return polys  

#max面積ｓ
def get_max_S(label_list):
    menseki = []
    for label in label_list:
        s = get_bnd_S(label)
        menseki.append(s)
    return int(max(menseki)),menseki

#面積別にする
#面積別に矩形を分ける
def get_s_bnd_precor(label_list,max_S):
    label_dic={}
    #最大値まで250刻みで格納
    for p,label in enumerate(label_list):
        for d,labe in enumerate(label):
            for minn in range(0,max_S,args.menseki_threshold):
                if minn < int(labe['menseki']) <= minn+args.menseki_threshold:
                    na = '%d' % (minn)
                    if na in label_dic:
                        label_dic[na].append(labe)
                    else:
                        label_dic[na] = [labe]
    #あぶれ回収
    for minn in range(0,max_S,args.menseki_threshold):
        na = '%d' % (minn)
        x = na in label_dic
        if x == False:
            r={}
            b = {}
            c = []
            r['category'] = 'space'
            b['category'] = 'space2'
            c.append(r)
            label_dic[na] = [r]

    return label_dic

#面積別に矩形を分ける
def get_s_bnd(label_list,max_S):
    label_dic={}
    #最大値まで250刻みで格納
    for p,label in enumerate(label_list):
        for d,labe in enumerate(label):
            for minn in range(0,max_S,args.menseki_threshold):
                if minn < int(labe[0]['boxA']['menseki']) <= minn+args.menseki_threshold:
                    na = '%d' % (minn)
                    if na in label_dic:
                        label_dic[na].append(labe)
                    else:
                        label_dic[na] = [labe]
    #あぶれ回収
    for minn in range(0,max_S,args.menseki_threshold):
        na = '%d' % (minn)
        x = na in label_dic
        if x == False:
            r={}
            b = {}
            c = []
            r['category'] = 'space'
            b['category'] = 'space2'
            c.append({'boxA':r,
                        'boxB':b})
            label_dic[na] = [c]

    return label_dic

#面積別に矩形を分ける
def get_s_bnd_2(label_list,max_S):
    label_dic={}
    #最大値まで250刻みで格納
    for p,label in enumerate(label_list):
        for d,labe in enumerate(label):
            for minn in range(0,max_S,args.menseki_threshold):
                if minn < int(labe['menseki']) <= minn+args.menseki_threshold:
                    na = '%d' % (minn)
                    if na in label_dic:
                        label_dic[na].append(labe)
                    else:
                        label_dic[na] = [labe]
    #あぶれ回収
    for minn in range(0,max_S,args.menseki_threshold):
        na = '%d' % (minn)
        x = na in label_dic
        if x == False:
            r={}
            b = {}
            c = []
            r['category'] = 'space'
            b['category'] = 'space2'
            c.append({'boxB':b})
            label_dic[na] = [c]

    return label_dic

def get_boxnum_s(label_list, categories,max_menseki):
    boxnum_list = {}

    for category in categories:
        for minn in range(0,int(max_menseki),args.menseki_threshold):
            if str(minn) in label_list:
                cnt = [x[0]['boxA']['category'] for x in label_list[str(minn)]].count(category)
                if str(minn) in boxnum_list:
                    boxnum_list[str(minn)].append([category, cnt])
                else:
                    if category=='field':
                        ss = []
                        ss.append([category, cnt])
                        boxnum_list[str(minn)] = ss
                    else:
                        boxnum_list[str(minn)] = [category, cnt]

    return boxnum_list

def get_boxnum_precor(label_list, categories,max_menseki):
    boxnum_list = {}

    for category in categories:
        for minn in range(0,int(max_menseki),args.menseki_threshold):
            if str(minn) in label_list:
                cnt = [x['category'] for x in label_list[str(minn)]].count(category)
                if str(minn) in boxnum_list:
                    boxnum_list[str(minn)].append([category, cnt])
                else:
                    if category=='field':
                        ss = []
                        ss.append([category, cnt])
                        boxnum_list[str(minn)] = ss
                    else:
                        boxnum_list[str(minn)] = [category, cnt]

    return boxnum_list

def cmi(obje1,obje2):
    for a,boxA in enumerate(obje1):
        for p,ob1 in enumerate(boxA):
            boxA_S = get_bnd_S(ob1)
    for b,boxB in enumerate(obje2):
        for ob2 in boxB:
            boxB_S = get_bnd_S(ob2)

"""
---------------------------------------------------------------------------------------------------

平均IOUを計算

---------------------------------------------------------------------------------------------------
"""
def calculate_averageiou(categories, detection):
    avg_iou = []

    for i, category in enumerate(categories):
        div = 0
        sum_iou = 0

        for dct in detection:
            if (dct[0]['boxA']['category'] == dct[0]['boxB']['category'] and category == dct[0]['boxA']['category']):
                div += 1
                sum_iou += dct[0]['iou']

        # 平均を算出
        if (div == 0):
            avg = 0

        else:
            avg = '{:.2f}'.format(sum_iou/div)

        avg_iou.append(avg)

    return avg_iou

"""
---------------------------------------------------------------------------------------------------

再現率を計算

---------------------------------------------------------------------------------------------------
"""
def calculate_recall(count_eval):
    recalls= []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if j <= size:
                Sum += int(num)
        if Sum != 0:
            recall = int(box[1+i]) / Sum 
            recalls.append(recall)
        else:
            recalls.append(0.0)

    return recalls

"""
---------------------------------------------------------------------------------------------------

適合率を計算

---------------------------------------------------------------------------------------------------
"""
def calculate_precisions(count_eval):
    precisions= []
    size = len(categories)
    
    for i, box in enumerate(count_eval):
        Sum = 0
        for j, num in enumerate(box):
            if  size < j < (size+1) * 2:
                Sum += int(num)
        if Sum != 0:
            precision = int(box[size+1+i]) / Sum 
            precisions.append(precision)
        else:
            precisions.append(0)

    return precisions

#F値計算
def fmeasure_cul(recalls,precisions,max_menseki_cor):
    f = {}
    #F値
    for minn in range(0,int(max_menseki_cor),args.menseki_threshold):
        ff = []
        for i in range(3):
            if  recalls[str(minn)][i] != 0 and precisions[str(minn)][i] != 0:
                fmeasure = 2 * recalls[str(minn)][i] * precisions[str(minn)][i] / ( recalls[str(minn)][i] + precisions[str(minn)][i] )
            else:
                fmeasure = 0
            ff.append(fmeasure)
        f[str(minn)]=ff
    return f

#csv
def write_csv(rotate,name,max_menseki_cor):
    save_row = {}
    save_row2 = {}
    with open(os.path.join(args.input_dir2,'%s.csv' % name),'w') as f:
        fieldnames = ["name","field", "green_field",'greenhouse']
        writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(rotate.keys())
        length = len(rotate)

        for minn in range(0,int(max_menseki_cor),args.menseki_threshold):
            for j in range(3):
                k =rotate[str(minn)][j]
                save_row[cate(j)] = k
            save_row['name'] = str(minn)
            writer.writerow(save_row)

def write_csv_k(rotate,name,max_menseki_cor):
    save_row = {}
    save_row2 = {}
    with open(os.path.join(args.input_dir2,'%s.csv' % name),'w') as f:
        fieldnames = ["name","field", "green_field",'greenhouse']
        writer = csv.DictWriter(f, fieldnames=fieldnames,delimiter=",",quotechar='"')
        writer.writeheader()

        k1 = list(rotate.keys())
        length = len(rotate)

        for minn in range(0,int(max_menseki_cor),args.menseki_threshold):
            for j in range(3):
                k =rotate[str(minn)][j][1]
                save_row[cate(j)] = k
            save_row['name'] = str(minn)
            writer.writerow(save_row)

def cate(j):
    if j==0:
        return 'field'
    if j==1:
        return 'green_field'
    if j==2:
        return 'greenhouse'

if __name__ == '__main__':
    #ファイルパスの取得
    xml_path_list,xml_path_list2 = import_data()
    #カテゴリー
    categories = read_category()
    #正解矩形の取得
    #obje1 = get_bbox_polygon(xml_path_list)
    obje1 = get_bbox(xml_path_list)
    #空白を削除
    obje1 = [i for i in obje1 if len(i)!=0]

    #推論結果矩形の取得
    #obje2 = get_bbox_polygon(xml_path_list2)
    obje2 = get_bbox(xml_path_list2)
    #空白を削除
    obje2 = [i for i in obje2 if len(i)!=0]
    cmi(obje1,obje2)
    #未検出(一つも推論ボックスとかぶらなかった正解ボックス)と検出(推論ボックスからみた正解ボックス)の取得
    notdetection_box_list,detection_box_list2,unnecessary_list,max_boxA,max_boxB = get_dete_notdete_box(categories,obje1,obje2)
    #アノテーションにダブりがあるため
    #pdb.set_trace()
    #空白を削除
    #notdetection_box_list = [i for i in notdetection_box_list if len(i)!=0]
    #材料
    #detection_box_list2 = [i for i in detection_box_list2 if len(i)!=0]
    #unnecessary_list = [i for i in unnecessary_list if len(i)!=0]
    #notdetection_box_list2 = get_dete_notdete_box(categories,notdetection_box_list,obje2)
    #面積別にする
    cor_menseki = get_s_bnd_precor(obje1,max_boxA)
    pre_menseki = get_s_bnd_precor(obje2,max_boxA)
    #detection_menseki = get_s_bnd(detection_box_list2,max_boxA)
    #notdetection_menseki = get_s_bnd_2(notdetection_box_list,max_boxA)
    #unnecessary_menseki = get_s_bnd(unnecessary_list,max_boxA)
    #detection_menseki_num = get_boxnum_s(detection_menseki, categories,max_boxA)
    cor_menseki_num = get_boxnum_precor(cor_menseki, categories,max_boxA)
    pre_menseki_num = get_boxnum_precor(pre_menseki, categories,max_boxA)
    liieval = {}
    '''
    #面積別に評価ごとに数をカウント
    for minn in range(0,int(max_boxA),args.menseki_threshold):
        count_eval = count_evaluation_num(categories, pre_menseki_num[str(minn)], cor_menseki_num[str(minn)], detection_menseki[str(minn)], unnecessary_menseki[str(minn)])
        liieval[str(minn)] = count_eval

    liiavg = {}
    #面積別に平均IOU
    for minn in range(0,int(max_boxA),args.menseki_threshold):
        avg_iou = calculate_averageiou(categories, detection_menseki[str(minn)])
        liiavg[str(minn)] = avg_iou
    
    looreca = {}
    #面積別に再現率
    for minn in range(0,int(max_boxA),args.menseki_threshold):
        recalls = calculate_recall(liieval[str(minn)])
        looreca[str(minn)] = recalls
    loopre = {}
    #面積別に適合率
    for minn in range(0,int(max_boxA),args.menseki_threshold):
        precisions = calculate_precisions(liieval[str(minn)])
        loopre[str(minn)] = precisions

    #F値
    f = fmeasure_cul(looreca,loopre,max_boxA)
    '''
    write_csv_k(cor_menseki_num,'num',max_boxA)
    #write_csv(loopre,'precisions',max_boxA)
    #write_csv(f,'fmeasure',max_boxA)

