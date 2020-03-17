from lxml import etree
import cv2
import annotation_io as anno
import random
import numpy as np
import os
import glob
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', default = './category.txt')
parser.add_argument('-x', '--xml_dir',  default = '/home/hiroshi/Chainer_Mask_R-CNN/dataset_mask/val')
parser.add_argument('-o', '--out_dir',  default = './val-draw')
args = parser.parse_args()




def draw():
    xml_path_list = glob.glob(os.path.join(args.xml_dir, 'label', '*.xml'))
    xml_path_list = sorted(xml_path_list)
    for ann in xml_path_list:
        fname = os.path.basename(ann)
        an = anno.AnnotationIO(ann)
        bboxes = anno.bboxes(an)
        label_name = get_label(ann)
        count = 0
        img = cv2.imread('/home/hiroshi/Chainer_Mask_R-CNN/dataset_mask/val/image/{}.jpg'.format(fname[:-4]))
        #元画像コピー
        dst = img.copy()

        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        color = (255,0,0)
        polygons = anno.polygons(an)
        
        for polygon in polygons:
            pts = np.array([polygon])
            pts = pts.reshape((-1,1,2))
            #alpa(pts,img)            
            #alpa(pts,img)
            cv2.polylines(img,[pts],True,(255,255,255),5)
            #cv2.fillConvexPoly(img,pts,(255,0,205))
            #cv2.fillConvexPoly(img,pts,(0,250,205))
            #label_name[count] == 'green_field':
            #cv2.fillConvexPoly(img,pts,(255,250,205))
            if label_name[count] == 'green_field':
                cv2.fillConvexPoly(img,pts,(230,23,230))
            if label_name[count] == 'field':
                cv2.fillConvexPoly(img,pts,(0,250,205))
            if label_name[count] == 'greenhouse':
                cv2.fillConvexPoly(img,pts,(225,105,65))
            count += 1
        count = 0
        
        #矩形表示
        #for b in bboxes:
        #    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 1)
        #    cv2.putText(img,label_name[count],(b[0], b[1]+30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 1, cv2.LINE_AA)
        #    count += 1
        
        #ポリゴンを透過して表示する
        img_final = cv2.addWeighted(dst,0.7,img,0.3,0,dst)
        #書き込み保存
        cv2.imwrite('./val-draw/'+fname[:-4]+".jpg", img_final)
        img = None
        an.objects = []
        
def get_label(xml_path):
    label_list = []

    tree = ET.parse(xml_path)
    filename = tree.find('filename').text

    for object in tree.findall('object'):

        labelname = object.find('category').find('value').text

        label_list.append(labelname)

    return label_list

if __name__ == '__main__':
    draw()
