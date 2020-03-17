import sys
import argparse
import glob
import random
import os
from PIL import Image
import copy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './233900_96200')
parser.add_argument('-ii', '--input_dir2',  default = './full_mask/mask')
parser.add_argument('-o', '--out_dir',  default = './output')
args = parser.parse_args()

#パスの読み込み
def import_data():
    img_path_list = glob.glob(os.path.join(args.input_dir, '**' ,'*.jpg'))
    img_path_list2 = glob.glob(os.path.join(args.input_dir2,'*.jpg'))
    img_path_list = sorted(img_path_list)
    img_path_list2 = sorted(img_path_list2)

    return img_path_list,img_path_list2

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

#マスク処理
def mask_img(img_path,dst):
    # Load two images
    img1 = cv2.imread(img_path,1)
    img2 = cv2.imread(dst,0)
    #マスク処理
    img_masked = cv2.bitwise_and(img1, img1, mask=img2 )
    #保存
    directory,framename = path_to_name(img_path)
    create_dir(directory)
    save = os.path.join(args.out_dir,directory,framename + '.jpg')
    cv2.imwrite(save, img_masked)

if __name__ == '__main__':

    # 読み込み
    img_path_list,img_path_list2 = import_data()
    lis = []

    #切り抜き
    for (img_path,liss) in zip(img_path_list,img_path_list2):
        #read_img(img_path,img_path_lis)
        mask_img(img_path,liss)