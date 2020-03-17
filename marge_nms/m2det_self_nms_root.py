# -*- coding: utf-8 -*-

# pylint: disable = C0111, C0301, C0325, C0326, C0103, C0304, C0303
# pylint: disable = E1310
# pylint: disable = W0611, W0512, W0612, W0622, W0621, W0613


import xml.etree.ElementTree as ET
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import argparse
import sys
import numpy as np
import colorsys
import random
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', default = './category.txt')
parser.add_argument('-p', '--pre_xml',  default = './data/val-MaskRCNN')
parser.add_argument('-d', '--dir_name', default = 'marge0.6')
parser.add_argument('-s', '--save_dir', default = 'draw-nms-0.6')
parser.add_argument('-i', '--draw_img', default = '/home/hiroshi/Dataset/folda-change/20191011矩形最終/val')
parser.add_argument('-se', '--self_or_normal', default = 'normal')
parser.add_argument('-o', '--out_dir',  default = './output-nms/val-MaskRCNN/draw-nms-0.6')
args = parser.parse_args()


"""
--------------------------------------------------------------------------------------------

カテゴリを読み込み

--------------------------------------------------------------------------------------------
"""
def read_category():
    # カテゴリの読み込み
    with open(args.category, 'r') as f:
        CLASSES = np.array(f.read().splitlines())

    return CLASSES


"""
--------------------------------------------------------------------------------------------

カテゴリごとに色を生成

--------------------------------------------------------------------------------------------
"""
def get_colors_for_classes(num_cat):
    hsv_tuples = [(x / num_cat, 1., 1.) for x in range(num_cat)]
    
    colors = list(map(lambda x:colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x:(int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
    #random.seed(10101)
    #random.shuffle(colors)
    #random.seed(None)


    return colors



"""
--------------------------------------------------------------------------------------------

XMLファイルからラベル情報を取得

--------------------------------------------------------------------------------------------
"""
def get_xmlbox(xml_path, category, height, width):
    box_list = []

    tree = ET.parse(xml_path)

    for object in tree.findall('object'):
        labelname = ''

        for cat in object.findall('category'):
            for i in category:
                if (i.upper() == cat.find('value').text.upper()):
                    labelname = cat.find('value').text

        score = object.find('score').text

        for bndbox in object.findall('bndbox'):
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text

            xmin = int(xmin) + width
            ymin = int(ymin) + height
            xmax = int(xmax) + width
            ymax = int(ymax) + height

            box_list.append([labelname, xmin, ymin, xmax, ymax, float(score)])

    # 昇順にソート
    box_list.sort()

    return box_list


"""
--------------------------------------------------------------------------------------------

NMS

--------------------------------------------------------------------------------------------
"""
def non_max_suppression(all_box_list, overlap_thresh):
    # scoreで降順にソート
    all_box_list = sorted(all_box_list, key=lambda x: x[5])

    # 内包した矩形を除外
    remove_idx_list = []

    for idx in tqdm(range(len(all_box_list))):
        i = all_box_list[idx]
        time.sleep(0.000001)

        for j in all_box_list:
            if (i != j):
                area_i = (int(i[3]) - int(i[1]) + 1) * (int(i[4]) - int(i[2]) + 1)
                area_j = (int(j[3]) - int(j[1]) + 1) * (int(j[4]) - int(j[2]) + 1)

                iw = min(int(j[3]), int(i[3])) - max(int(j[1]), int(i[1])) + 1
                ih = min(int(j[4]), int(i[4])) - max(int(j[2]), int(i[2])) + 1

                if (iw > 0 and ih > 0):
                    ans_i = iw*ih / area_i
                    ans_j = iw*ih / area_j

                    if (ans_i > ans_j and ans_i >= overlap_thresh):
                        remove_idx_list.append(idx)

    keep_box_list = [all_box_list[k] for k in range(len(all_box_list)) if not k in remove_idx_list]

    return keep_box_list

def get_scores(boxes):
    scor = []
    box = []
    cate = []
    for i in boxes:
        scor.append(i[5])
    for j in boxes:
        box.append([j[1],j[2],j[3],j[4]])
    for n in boxes:
        cate.append(n[0])

    
    return scor,box,cate


def non_max_suppression_z(boxes,overlap_thresh):
    '''Non Maximum Suppression (NMS) を行う。

    Args:
        boxes     : (N, 4) の numpy 配列。矩形の一覧。
        overlap_thresh: [0, 1] の実数。閾値。

    Returns:
        boxes : (M, 4) の numpy 配列。Non Maximum Suppression により残った矩形の一覧。
    '''
    #合わせ
    s,box,c = get_scores(boxes)
    boxe = np.array(box)
    s = np.array(s)
    c = np.array(c)
    if boxe.size == 0:
        return []
    if len(boxe) <= 1:
        return boxes
    # float 型に変換する。
    boxe = boxe.astype("float")
    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxe, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(s)  # スコアを降順にソートしたインデックス一覧
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
    #scores = np.array(scores)
    #cate = np.array(cate)
    #boxes = np.array(boxes)
    bb = []
    for iii in selected:
        bb.append(boxes[iii])


    return bb#,scores[selected],cate[selected]

"""
--------------------------------------------------------------------------------------------

画像に矩形を描画

--------------------------------------------------------------------------------------------
"""
def draw_box(imgpath, savepath, rectbox_list, category, colors):
    fnt = ImageFont.truetype('./font/FiraMono-Medium.otf', 15)

    img = Image.open(imgpath)
    draw = ImageDraw.Draw(img)

    for box in rectbox_list:
        for i, cat in enumerate(category):
            if (box[0].upper() == cat.upper()):
                draw.line(((box[1], box[2]), (box[3], box[2])), fill=colors[i], width=2) # 上
                draw.line(((box[1], box[4]), (box[3], box[4])), fill=colors[i], width=2) # 下
                draw.line(((box[1], box[2]), (box[1], box[4])), fill=colors[i], width=2) # 左
                draw.line(((box[3], box[2]), (box[3], box[4])), fill=colors[i], width=2) # 右

                #x = int(((box[3] - box[1]) / 2) + box[1])
                #y = int(((box[4] - box[2]) / 2) + box[2])

                #draw.ellipse((x-7, y-7, x+7, y+7), fill=colors[i]) 

                #draw.text((box[3]+5, box[4]-30), cat, font=fnt, fill=colors[i])
                draw.text((box[3]+5, box[4]-15), '{:.2f}'.format(box[5]), font=fnt, fill=colors[i])

    img.save(savepath)


"""
--------------------------------------------------------------------------------------------

xmlファイルを作成

--------------------------------------------------------------------------------------------
"""
def create_xml(framename,directory, box_list):
    root   = ET.Element('annotation')
    filename = ET.SubElement(root, 'filename')
    filename.text = framename

    for d in box_list:
        obj    = ET.SubElement(root, 'object')
        bndbox = ET.SubElement(obj, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')

        xmin.text = str(d[1])
        ymin.text = str(d[2])
        xmax.text = str(d[3])
        ymax.text = str(d[4])

        score = ET.SubElement(obj, 'score')
        score.text = '{:.2f}'.format(d[5])

        category = ET.SubElement(obj, 'category')
        value    = ET.SubElement(category, 'value')
        value.text = d[0]

    save_dir = os.path.join(args.out_dir,directory,'supervised')

    # ディレクトリを作成
    if not (os.path.isdir(save_dir)):
        os.makedirs(save_dir)
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(os.path.join(save_dir,framename + '.xml'))

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
    # カテゴリを読み込み
    category = read_category()

    # カテゴリごとに色を生成
    colors = get_colors_for_classes(len(category))

    # 入力画像を取得
    img_list = glob.glob(os.path.join(args.draw_img, '**', '*.jpg'))

    # ディレクトリを作成
    if not (os.path.isdir(os.path.join(args.out_dir))):
        os.makedirs(os.path.join(args.out_dir))

    # 推論ボックスを描画
    for img_path in img_list:
        directory = img_path.rsplit('/', 2)[1]
        framename = img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        clop_xml_list = glob.glob(os.path.join(args.pre_xml, args.dir_name, directory, 'supervised' ,framename + '*.xml'))

        all_box_list = []

        for clop_xml in clop_xml_list:
            clop_frame = clop_xml.rsplit('/', 1)[1].rsplit('.', 1)[0]
            height = int(clop_frame.rsplit('_', 2)[1])
            width  = int(clop_frame.rsplit('_', 2)[2])
            #height = int(clop_frame.rsplit('_', 2)[2])*512
            #width  = int(clop_frame.rsplit('_', 2)[1])*512

            # xmlboxを取得
            box_list = get_xmlbox(clop_xml, category, height, width)
            all_box_list.extend(box_list)
        if args.self_or_normal == 'self':
            selected_box_list = non_max_suppression(all_box_list, overlap_thresh=0.8)
        else:
            selected_box_list = non_max_suppression_z(all_box_list, overlap_thresh=0.8)

        # ディレクトリを作成
        if not (os.path.isdir(os.path.join(args.out_dir, directory))):
            os.makedirs(os.path.join(args.out_dir, directory))

        # 保存先を設定
        output_path = os.path.join(args.out_dir, directory, framename + '.jpg')

        # xmlboxを描画
        draw_box(img_path, output_path, selected_box_list, category, colors)

        # xmlファイル作成
        create_xml(framename,directory, selected_box_list)

    print ("Successful Completion Predict !!")
