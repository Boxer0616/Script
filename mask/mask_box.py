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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category',      default = './category2.txt')
parser.add_argument('-i', '--input_dir',  default = './dataset/full_mask/mask')
parser.add_argument('-ii', '--input_dir2',  default = './dataset/ketugoukekka_nms/ketubgou_nms/xml')
parser.add_argument('-o', '--out_dir',  default = './output')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir, '*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir2, '**' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)
    
    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 2)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory))):
        os.makedirs(os.path.join(args.out_dir, directory))

#保存パス
def get_save_path(save_dir,file_idx,bdb):
    
    path = os.path.join(save_dir,"%s_%04d_%04d_%04d_%04d.jpg" % (file_idx, bdb['xmin'], bdb['ymin'],bdb['xmax'], bdb['ymax']))

    return path

#黒を透過
def black_schelton(img_path2):
    file_name = img_path2
    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
    directory,framename = path_to_name(img_path2)
    #blended = cv2.addWeighted(src1=img1,alpha=0.5,src2=img2,beta=0.5,gamma=0)
    
    save = os.path.join('./output_bb',framename + '.jpg')
    cv2.imwrite(save, dst)

    return save

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
    dx = xB - xA
    dy = yB - yA
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
    if dx>0 and dy>0:
        return iou
    else:
        return 0

#検出
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
    for i, s in enumerate(iou_list[:]):
        for x in s:
            for j, z in enumerate(iou_list[:]):
                for y in z:
                    if (x['boxA']['filename'] == y['boxA']['filename'] and x['boxA']['boxlabel'] != y['boxA']['boxlabel'] and x['boxB']['boxlabel'] == y['boxB']['boxlabel']):
                        # iouが小さい方を削除
                        if (x['iou'] >= y['iou']):
                            iou_list.remove(z)

                        elif (x['iou'] ==y['iou']):
                            if (x['boxB']['score'] >= y['boxB']['score'] and y['boxA']['category'] != y['boxB']['category']):
                                iou_list.remove(z)

                            elif (x['boxA']['category'] == x['boxB']['category']):
                                iou_list.remove(z)

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

#検出
def detection_box(boA, boB):
    # determine the (x, y)-coordinates of the intersection rectangle
    dete = []
    detection_box = []
    notdetection_box = []
    #正解boxに推論box総当りでIou計算
    for a,boxA in enumerate(boA):
        dete = []
        not_dete = []
        for b,boxB in enumerate(boB):
            iou = bb_intersection_over_union(boxA,boxB)
            if iou != 0:
                dete.append({'iou':iou,
                             'boxA':boxA,
                             'boxB':boxB})
        #Iouが重なっている正解ボックスと推論ボックスを格納
        detection_box.append(dete)
        if len(detection_box[a]) == 0:
            #一つも重なる推論ボックスがなかった正解ボックスを格納
            notdetection_box.append({'boxA':boxA,})
    #空白を削除
    dd = [i for i in detection_box if len(i)!=0]
		
    return dd,notdetection_box


#未検出矩形を取得
def get_miss_box(detection,ob1):
    ob1_list = ob1.tolist()
    for c,obbe in enumerate(ob1_list[:]):
        for obj in detection:
            if obbe['boxlabel'] == obj[0]['boxA']['boxlabel']:
                ob1_list.remove(obbe)
    
    return ob1_list

#未検出ボックス取得
def get_dete_notdete_box(categories,obje1,obje2):

    not_detection_box_list = []

    for p,(ob1,ob2) in enumerate(zip(obje1,obje2)):
        #ボックスの数
        #corbox_num_list = get_boxnum(ob1,categories)
        #prebox_num_list = get_boxnum(ob2,categories)
        #しきい値以上のIouをもつボックスの組み合わせを返す
        detection_b,not_detection_box = detection_box(ob1,ob2)
        #detection_box,not_detection_box = detection_or_not_detection(ob1, ob2)
        #ダブリを削除、不要を抽出
        detection,unnecessary = classify_iou(detection_b)
        #未検出矩形を取り出す
        miss_box = get_miss_box(detection,ob1)
        not_detection_box_list.append(miss_box)

    return not_detection_box_list


# カテゴリの読み込み
def read_category():
    with open(args.category, 'r') as f:
        categories = f.read().splitlines()
    
    return categories

#マスク部分の輪郭を返す
def find_contor(img_path):
    #グレースケールで読み込み
    img = cv2.imread(img_path,0)
    #img2 = cv2.imread(img_path)
    ret,thresh = cv2.threshold(img,127,255,0)
    #輪郭抽出
    imgEdge,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #plt.imshow(img2)
    #plt.show()
    return contours,img_path

#外接矩形を返す
def boundingRect(contours):
    bounding = []
    count = 0
    der,name = path_to_name(contours[1])
    for cnt in contours[0]:
        # 輪郭に外接する長方形を取得する。
        x, y, width, height = cv2.boundingRect(cnt)
        #外接矩形2点
        bounding.append({'xmin':x,'ymin':y,'xmax':x+width,'ymax':y+height,'boxlabel':str(count)+ '_' + name,'filename':name,'category':'field'})
        #確認表示
        #test_display(contours[1],x,y,width,height)
        count += 1
    
    return bounding

#確認用
def test_display(contours,x,y,width,height):
    #描画用
    img2 = cv2.rectangle(contours,(x,y),(x+width,y+height),(0,255,0),2)
    plt.imshow(img2)
    plt.show()

#マスクの輪郭を抽出して外接矩形を返す
def get_bdb_box_mask(img_path_list):
    contours = []
    bound = []
    for img_path in img_path_list:
        contours.append(find_contor(img_path))
    for counto in contours:
        bound.append(boundingRect(counto))
    bouy = np.array(bound)
    return bouy

if __name__ == '__main__':
    #ファイルパスの取得
    img_path_list,xml_path_list = import_data()
    #カテゴリー
    categories = read_category()
    #矩形の取得
    obje = get_bbox(xml_path_list)
    #マスクの外接矩形を返す
    bdb = get_bdb_box_mask(img_path_list)
    #マスクと重ならない矩形を取得
    #notdetection_box_list = get_dete_notdete_box(categories,obje,bdb)
    #xmlに書き込み
    for pp,obb in enumerate(bdb):
        #xml作成
    	create_xml(obb)