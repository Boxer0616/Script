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
parser.add_argument('-i', '--input_dir',  default = './data/val-MaskRCNN/remove-edge')
parser.add_argument('-o', '--out_dir',  default = './data/val-MaskRCNN/marge0.6')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join('../color_change/mask_clip_clop','**','*.jpg'))
    img_path_list = sorted(img_path_list)
    xml_path_list = glob.glob(os.path.join(args.input_dir, '**' ,'supervised' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)

    return img_path_list,xml_path_list

#ファイル名とディレクトリ名
def path_to_name(data_path):
    directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

    return directory,framename

#矩形を取得
def get_bbox_polygon(xml_path_list):
    objects = []
    obj_cnt = []
    objj = []
    scores = []
    catego = []
    name = []
    cnt = 0
    #フォルダごとにループ
    for xml_path in xml_path_list:
        objj_0 = []
        sc_0 = []
        nn_0 = []
        cc_0 = []
        #結合後が同じ画像ごとにループ
        for nn,xm in enumerate(xml_path):
            ##1ファイルずつループ 
            objj,sc,nn,cc = get_bbox(xm,obj_cnt,cnt)
            
            objj_0.append(objj)
            sc_0.append(sc)
            nn_0.append(nn)
            cc_0.append(cc)
        
        objects.append(objj_0)
        scores.append(sc_0)
        catego.append(cc_0)
        name.append(nn_0)

    del obj_cnt[0]
    return objects,scores,catego,name

#1ファイルずつ矩形を取得
def get_bbox(xm,obj_cnt,cnt):         
    objj = []
    sc = []
    nn = []
    cc = []
    for n, xml in enumerate(xm):
        tree = etree.parse(xml)
        filename = tree.find('filename').text
        obj_cnt.append(cnt)
        cnt = 0
        objs = tree.xpath('//object')
        for obj in objs:
            bdbs = obj.xpath('bndbox') 
            for bdb in bdbs:
                objj.append({
                0 : int(bdb.xpath('.//xmin')[0].text),
                1 : int(bdb.xpath('.//ymin')[0].text),
                2 : int(bdb.xpath('.//xmax')[0].text),
                3 : int(bdb.xpath('.//ymax')[0].text)
                })
                nn.append(filename)
            scoree = obj.xpath('score')
            for scor in scoree:
                sc.append(scor.text)
            cate = obj.xpath('.//category/value')
            for scr in cate:
                cc.append(scr.text)
    return objj,sc,nn,cc


#結合後の矩形を取得
def after_bbox(objects,name):
    #フォルダごと
    for ll,na in enumerate(name):
        #結合後が同じ画像ごと
        for ee,bbb in enumerate(na):
            #1ファイルずつ
            for e,bb in enumerate(bbb):
                nn = bb.rsplit("_")
                x = int(nn[5])
                y = int(nn[4])
                culculate_bbox(objects[ll][ee][e],x,y)

    return objects

#結合後の矩形
def culculate_bbox(obje,x,y):
    if x == 0 and y != 0:
        obje[1] = obje[1]+y
        obje[3] = obje[3]+y
    if x != 0 and y == 0:
        obje[0] = obje[0]+x
        obje[2] = obje[2]+x
    if x != 0 and y != 0:
        obje[1] = obje[1]+y
        obje[3] = obje[3]+y 
        obje[0] = obje[0]+x
        obje[2] = obje[2]+x

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

xmlファイルを作成

--------------------------------------------------------------------------------------------
"""
def create_xml(n, obj,scores,cate,namee):
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
    tree.write(args.out_dir + '/' + directory + '/' + 'supervised' + '/' + file  + '.xml', encoding="utf-8", xml_declaration=True)


#ディレクトリ作成
def create_dir(directory):
    # ディレクトリがなければ作成
    if not (os.path.isdir(os.path.join(args.out_dir, directory,'supervised'))):
        os.makedirs(os.path.join(args.out_dir, directory,'supervised'))


"""
--------------------------------------------------------------------------------------------

xmlファイル書き込む

--------------------------------------------------------------------------------------------
"""
def writeXml(xmlRoot, path):
    
    encode = "utf-8"
    
    xmlFile = open(path, "w")
    
    document = md.parseString(ET.tostring(xmlRoot, encode))
    
    document.writexml(
        xmlFile,
        encoding = encode,
        newl = "\n",
        indent = "",
        addindent = "  "
    )

#画像を結合
def img_ketsugou(img_path_list):
    img = zip(*[iter(img_path_list)]*2)
    
    for n,imn in enumerate(img):
        img1 = cv2.imread(imn[0])
        img2 = cv2.imread(imn[1])
        im_h = cv2.hconcat([img1, img2])
        cv2.imwrite(args.out_dir + '/' + 'ketsugou' + '_' + str(n) + '.jpg',im_h)

#xmlフォルダで分ける
def xml_data_ketsugou(xml_path_list):
    t_aml = []
    xmm = []
    na = []
    dirc = []
    l = len(xml_path_list)
    for n,xml_path in enumerate(xml_path_list):

        directory,framename = path_to_name(xml_path)
        if n+1<l:
            directory2,framename2 = path_to_name(xml_path_list[n+1])
        else:
            directory2 = 0
        #ディレクトリ名が同じならまとめる
        if directory == directory2:
            xmm.append(xml_path)
            na.append(framename)
        else:
            xmm.append(xml_path)
            t_aml.append(xmm)
            xmm = []
            dirc.append(na)
            na = []
    #結合後が同じ画像ごとにまとめる
    matomee,name = xml_matome(t_aml,dirc)
    return matomee,name

#結合後が同じ画像ごとにまとめる
def xml_matome(t_aml,dirc):
    l = len(t_aml)
    aaa = []
    matome = []
    matome_name = []
    nam = []
    for p,ttt in enumerate(t_aml):
        fi = []
        for pp,tt in enumerate(ttt):
            l = len(ttt)
            nn = tt.rsplit("_")
            if pp+1<l:
                nnn = ttt[pp+1].rsplit("_")
            else:
                nnn = copy.copy(nn)
                nnn[3] = 'da'
                nnn[4] = 'my'
            file = nn[3]+'_'+nn[4]
            file2 = nnn[3]+'_'+nnn[4]
            #ファイル名（同じ画像を示す部分）が同じならまとめる
            if file == file2:
                aaa.append(tt)
            else:
                aaa.append(tt)
                fi.append(aaa)
                aaa = []
        matome.append(fi)
    #名前も同様に
    l = len(dirc)
    for p,ttt in enumerate(dirc):
        fi = []
        for pp,tt in enumerate(ttt):
            l = len(ttt)
            nn = tt.rsplit("_")
            if pp+1<l:
                nnn = ttt[pp+1].rsplit("_")
            else:
                nnn = copy.copy(nn)
                nnn[0] = 'da'
                nnn[1] = 'my'
                nnn[2] = 'da'
                nnn[3] = 'my'               
            file = nn[0] + '_' + nn[1] + '_' + nn[2] + '_' + nn[3]
            file2 = nnn[0] + '_' + nnn[1] + '_' + nnn[2] + '_' + nnn[3]
            if file == file2:
                aaa.append(file)
            else:
                aaa.append(file)
                fi.append(aaa)
                aaa = []
        matome_name.append(fi)

    return matome,matome_name




if __name__ == '__main__':
    #データのパス
    img_path_list,xml_path_list = import_data()
    #画像の結合
    #img_ketsugou(img_path_list)
    #結合後が同じ画像をフォルダでまとめる
    xml,namee = xml_data_ketsugou(xml_path_list)
    #矩形とポリゴンを取得
    objects,scores,catego,name = get_bbox_polygon(xml)
    #after矩形を取得
    after_obj = after_bbox(objects,name)
    #xmlに書き込み
    for nn,objg in enumerate(after_obj):
        for nnn,objj in enumerate(objg):
            create_xml(nnn,objj,scores[nn][nnn],catego[nn][nnn],name[nn][nnn])
