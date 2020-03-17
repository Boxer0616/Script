import sys
import argparse
import glob
import random
import os
from PIL import Image
import copy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir',  default = './predict')
parser.add_argument('-ii', '--input_dir2',  default = './machi_mask/tt')
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

#画像合成
def read_img(img_path,dst):
    # Load two images
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(dst)

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]
    #マスクの作成
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    
    '''
    マスク合成するか、マスク以外を切り抜くか
    '''
    #マスク部分の合成
    img1[0:rows, 0:cols ] = dst
    #マスク部分を切り抜き
    #img1[0:rows, 0:cols ] = img1_bg
    '''
    '''
    #保存
    directory,framename = path_to_name(img_path)
    create_dir(directory)
    save = os.path.join(args.out_dir,directory,framename + '.jpg')
    cv2.imwrite(save, img1)

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


if __name__ == '__main__':

    # 読み込み
    img_path_list,img_path_list2 = import_data()
    lis = []
    #透過
    for img_path2 in img_path_list2:
        lis.append(black_schelton(img_path2))
    lis = sorted(lis)
    #合成
    for (img_path,img_path_lis) in zip(img_path_list,lis):
        read_img(img_path,img_path_lis)
