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
from xml.dom.minidom import Document

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category',    default = './category.txt')
parser.add_argument('-i', '--input_dir', default = './output')
parser.add_argument('-o', '--output_dir', default = './out')
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

def save_to_xml(save_path, im_height, im_width, objects_axis, label_name):
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('x0')
        if (objects_axis[i][0] < 0):
            objects_axis[i][0] = 0
        elif (objects_axis[i][0] > 800):
            objects_axis[i][0] = 800
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('y0')
        if (objects_axis[i][1] < 0):
            objects_axis[i][1] = 0
        elif (objects_axis[i][1] > 800):
            objects_axis[i][1] = 800
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('x1')
        if (objects_axis[i][2] < 0):
            objects_axis[i][2] = 0
        elif (objects_axis[i][2] > 800):
            objects_axis[i][2] = 800
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('y1')
        if (objects_axis[i][3] < 0):
            objects_axis[i][3] = 0
        elif (objects_axis[i][3] > 800):
            objects_axis[i][3] = 800
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)
        
        x2 = doc.createElement('x2')
        if (objects_axis[i][4] < 0):
            objects_axis[i][4] = 0
        elif (objects_axis[i][4] > 800):
            objects_axis[i][4] = 800
        x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        bndbox.appendChild(x2)
        y2 = doc.createElement('y2')
        if (objects_axis[i][5] < 0):
            objects_axis[i][5] = 0
        elif (objects_axis[i][5] > 800):
            objects_axis[i][5] = 800
        y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        bndbox.appendChild(y2)

        x3 = doc.createElement('x3')
        if (objects_axis[i][6] < 0):
            objects_axis[i][6] = 0
        elif (objects_axis[i][6] > 800):
            objects_axis[i][6] = 800
        x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        bndbox.appendChild(x3)
        y3 = doc.createElement('y3')
        if (objects_axis[i][7] < 0):
            objects_axis[i][7] = 0
        elif (objects_axis[i][7] > 800):
            objects_axis[i][7] = 800
        y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        bndbox.appendChild(y3)

        object_cat = doc.createElement('category')
        objects.appendChild(object_cat)

        value = doc.createElement('value')
        value.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
        object_cat.appendChild(value)

    f = open(save_path,'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()


def clip_bbox(anno, x_from, y_from, w, h):

    ann = copy.deepcopy(anno)
    x_to = x_from+w
    y_to = y_from+h

    objs = ann.findall('object')

    #print(len(ann.findall('object')))
    for obj in objs:
        bndbox_anno = obj.find('bndbox')
        xmin = int(int(bndbox_anno.find('xmin').text))
        xmax = int(int(bndbox_anno.find('xmax').text))
        ymin = int(int(bndbox_anno.find('ymin').text))
        ymax = int(int(bndbox_anno.find('ymax').text))

        '''ポリゴン頂点数が2以下だったらおかしなデータなので削除する'''
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
    '''
    objsdd = ann.findall('object')
    for objsd in objsdd:
        obj_rote = objsd.find('.//bndbox')
        for i in range(4):
            if (int(obj_rote.find('x'+str(i)).text)-x_from) <=0:
                obj_rote.find('x'+str(i)).text = '0'
            else:
                obj_rote.find('x'+str(i)).text = str(int(obj_rote.find('x'+str(i)).text)-x_from)
            if (int(obj_rote.find('x'+str(i)).text)-x_from) >= w:
                obj_rote.find('x'+str(i)).text = str(w)
            else:
                obj_rote.find('x'+str(i)).text = str(int(obj_rote.find('x'+str(i)).text)-x_from)          
            if (int(obj_rote.find('y'+str(i)).text)-x_from) <= 0:
                obj_rote.find('y'+str(i)).text ='0'
            else:
                obj_rote.find('y'+str(i)).text = str(int(obj_rote.find('y'+str(i)).text)-y_from)
            if (int(obj_rote.find('y'+str(i)).text)-x_from) >= h:
                obj_rote.find('y'+str(i)).text = str(h)
            else:
                obj_rote.find('y'+str(i)).text = str(int(obj_rote.find('y'+str(i)).text)-y_from)
    '''    
    '''
        obj_rote.find('x0').text = str(int(obj_rote.find('x0').text)-x_from)
        obj_rote.find('y0').text = str(int(obj_rote.find('y0').text)-y_from)
        obj_rote.find('x1').text = str(int(obj_rote.find('x1').text)-x_from)
        obj_rote.find('y1').text = str(int(obj_rote.find('y1').text)-y_from)
        obj_rote.find('x2').text = str(int(obj_rote.find('x2').text)-x_from)
        obj_rote.find('y2').text = str(int(obj_rote.find('y2').text)-y_from)
        obj_rote.find('x3').text = str(int(obj_rote.find('x3').text)-x_from)
        obj_rote.find('y3').text = str(int(obj_rote.find('y3').text)-y_from)
    '''
    '''
        if False:
           # 一部が矩形をまたがっている場合と内包している場合
            xmin = max([xmin, x_from])
            ymin = max([ymin, y_from])
            xmax = min([xmax, x_to])
            ymax = min([ymax, y_to])
    '''
    for bn in ann.findall('./object/bndbox'):
        '''
        for i in range(4):
            bn.find('x'+str(i)).text = str(int(bn.find('x'+str(i)).text)-x_from)
            bn.find('y'+str(i)).text = str(int(bn.find('y'+str(i)).text)-y_from)
        re_xmin = bn.find('xmin')
        bn.remove(re_xmin)
        re_ymin = bn.find('ymin')
        bn.remove(re_ymin)
        re_xmax = bn.find('xmax')
        bn.remove(re_xmax)
        re_ymax = bn.find('ymax')
        bn.remove(re_ymax)
        '''
        for i in range(4):
            if (int(bn.find('x'+str(i)).text)-x_from) <=0:
                bn.find('x'+str(i)).text = '0'
            elif(int(bn.find('x'+str(i)).text)-x_from)<w:
                bn.find('x'+str(i)).text = str(int(bn.find('x'+str(i)).text)-x_from)
            if (int(bn.find('x'+str(i)).text)-x_from) >= w:
                bn.find('x'+str(i)).text = str(w)
            elif(int(bn.find('x'+str(i)).text)-x_from)>0:
                bn.find('x'+str(i)).text = str(int(bn.find('x'+str(i)).text)-x_from)

            if (int(bn.find('y'+str(i)).text)-y_from) <= 0:
                bn.find('y'+str(i)).text ='0'
            elif(int(bn.find('y'+str(i)).text)-y_from)<h:
                bn.find('y'+str(i)).text = str(int(bn.find('y'+str(i)).text)-y_from)
            if (int(bn.find('y'+str(i)).text)-y_from) >= h:
                bn.find('y'+str(i)).text = str(h)
            elif(int(bn.find('y'+str(i)).text)-y_from)>0:
                bn.find('y'+str(i)).text = str(int(bn.find('y'+str(i)).text)-y_from)
        re_xmin = bn.find('xmin')
        bn.remove(re_xmin)
        re_ymin = bn.find('ymin')
        bn.remove(re_ymin)
        re_xmax = bn.find('xmax')
        bn.remove(re_xmax)
        re_ymax = bn.find('ymax')
        bn.remove(re_ymax)
        

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
    '''
    objs3 = ann.findall('object')
    for objb in objs3:
        rotate_bnd = []
        rotate_bnds = []
        clip_polygon = []
        obj_dert = objb.find('./bndbox')
        x0 = int(obj_dert.find('x0').text)
        y0 = int(obj_dert.find('y0').text)
        x1 = int(obj_dert.find('x1').text)
        y1 = int(obj_dert.find('y1').text)
        x2 = int(obj_dert.find('x2').text)
        y2 = int(obj_dert.find('y2').text)
        x3 = int(obj_dert.find('x3').text)
        y3 = int(obj_dert.find('y3').text)
        rotate_bn = ((x0,y0),(x1,y1),(x2,y2),(x3,y3))
        #rotate_bnds.append(((60,-98),(26,-98),(30,-243),(65,-242)))
        rotate_bnds.append(rotate_bn)
        #print(polygons)
        area = cv2.contourArea(np.array(rotate_bnds))
        if area > 0:
            #ポリゴンをクリップ
            clip_polygon = clip_poly(clip,rotate_bnds)
        else:#面積が0だったらオブジェクトを削除
            objs3.remove(objb)
        print(rotate_bnds)
        print(clip_polygon)
        #ポリゴンがクリップされたら変更
        if len(clip_polygon) != 0:
            #new_rotate_bnd = ET.SubElement(obj_dert,'bndbox')
            for n_po in clip_polygon:
                if len(n_po)<4:
                    objs3.remove(objb)
                else:
                    for i in range(4):
                        obj_dert.remove(obj_dert.find('x'+str(i)))
                        obj_dert.remove(obj_dert.find('y'+str(i)))
                    if len(n_po) == 4:
                        new_x0 = ET.SubElement(obj_dert,'x0')
                        new_x0.text = str(n_po[0][0])
                        new_x1 = ET.SubElement(obj_dert,'x1')
                        new_x1.text = str(n_po[1][0])
                        new_x2 = ET.SubElement(obj_dert,'x2')
                        new_x2.text = str(n_po[2][0])
                        new_x3 = ET.SubElement(obj_dert,'x3')
                        new_x3.text = str(n_po[3][0])

                        new_y0 = ET.SubElement(obj_dert,'y0')
                        new_y0.text = str(n_po[0][1])
                        new_y1 = ET.SubElement(obj_dert,'y1')
                        new_y1.text = str(n_po[1][1])
                        new_y2 = ET.SubElement(obj_dert,'y2')
                        new_y2.text = str(n_po[2][1])
                        new_y3 = ET.SubElement(obj_dert,'y3')
                        new_y3.text = str(n_po[3][1])
                    else:
                        new_x0 = ET.SubElement(obj_dert,'x0')
                        new_x0.text = str(n_po[3][0])
                        new_x1 = ET.SubElement(obj_dert,'x1')
                        new_x1.text = str(n_po[4][0])
                        new_x2 = ET.SubElement(obj_dert,'x2')
                        new_x2.text = str(n_po[0][0])
                        new_x3 = ET.SubElement(obj_dert,'x3')
                        new_x3.text = str(n_po[1][0])

                        new_y0 = ET.SubElement(obj_dert,'y0')
                        new_y0.text = str(n_po[3][1])
                        new_y1 = ET.SubElement(obj_dert,'y1')
                        new_y1.text = str(n_po[4][1])
                        new_y2 = ET.SubElement(obj_dert,'y2')
                        new_y2.text = str(n_po[0][1])
                        new_y3 = ET.SubElement(obj_dert,'y3')
                        new_y3.text = str(n_po[1][1])
      

    '''
'''
def sa_dayo(n_pooo):
    a = np.array(n_pooo)
    for i in range(5):
        for jj in range(5):
            min = np.linalg.norm(a[i] - a[jj])
            if min >0:
                min_o = min
                result = np.array([min, i, j])
        for nj,j in enumerate(a):
            c = np.linalg.norm(i - j)
            if min > c and min>0 and c>0 :
                min = c
                #index_a = np.where(a[:] == i)
                #index_b = np.where(b[:] == j)
                result = np.array([min, i, j])
            else:
                pass
    return numm
    
    sa_list1 = []
    sa_list2 = []
    sa_list_y = []
    sa_list_y2 = []
    for i in range(5):
        for n in range(5):
            sa_x = n_pooo[i][0]-n_pooo[n][0]
            if sa_x >0:
                sa_list1.append(sa_x)
                sa_list2.append((sa_x,i,n))
    ss = sa_list1.index(min(sa_list1))
    min_ss = sa_list2[ss]
    #for saa in sa_list:

    for i in range(4):
        for n in range(5):
            sa_y = n_pooo[i][1]-n_pooo[n][1]
            if sa_y >0:
                sa_list_y.append(sa_y)
                sa_list_y2.append((sa_y,i,n))
    ss2 = sa_list_y.index(min(sa_list_y))
    min_ss_y = sa_list_y2[ss2]
'''

#ポリゴンクリップ
def clip_poly(clip,polygons):

    pc = pyclipper.Pyclipper()
    pc.AddPath(clip, pyclipper.PT_CLIP, True)
    pc.AddPaths(polygons, pyclipper.PT_SUBJECT, True)
    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

    return solution

def get_xmlbox(xml_path):
    box_list = []    
    tree = ET.parse(xml_path)
    for object in tree.findall('object'):
        labelname = object.find('name').text
        pose = object.find('pose').text
        truncated = object.find('truncated').text
        difficult = object.find('difficult').text
        for bndbox in object.findall('bndbox'):
            x0 = int(bndbox.find('x0').text)
            y0 = int(bndbox.find('y0').text)
            x1 = int(bndbox.find('x1').text)
            y1 = int(bndbox.find('y1').text)
            x2 = int(bndbox.find('x2').text)
            y2 = int(bndbox.find('y2').text)
            x3 = int(bndbox.find('x3').text)
            y3 = int(bndbox.find('y3').text)

            box_list.append([x0, y0, x1, y1, x2, y2, x3, y3])

    # 昇順にソート
    box_list.sort()

    return box_list

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
                    if not (os.path.isdir(os.path.join(args.output_dir, 'supervised'))):
                        os.makedirs(os.path.join(args.output_dir, 'supervised'))
                    cimg, cxml = clip_fix_data(f, start_w_new, start_h_new, xml)
                    cxml = clip_bbox(xml, start_w_new, start_h_new, int(args.size), int(args.size))
                    #img保存
                    save_img_path = os.path.join(args.output_dir,'image', "%s_%04d_%04d.jpg" % (file_name, start_h_new, start_w_new))
                    cimg.save(save_img_path)
                    #xml保存
                    save_xml_path = os.path.join(args.output_dir,'supervised', "%s_%04d_%04d.xml" % (file_name, start_h_new, start_w_new))
                    cxml.write(save_xml_path)
                    count += 1

if __name__ == '__main__':

    # 分割
    devide()
