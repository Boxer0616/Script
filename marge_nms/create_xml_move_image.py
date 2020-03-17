import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml_dir', default='./data/val-MaskRCNN/val')
args = parser.parse_args()


"""
---------------------------------------------------------------------------------------------------

メイン関数

---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    xml_path_list = glob.glob(os.path.join(args.xml_dir, '**' ,'*.xml'))
    xml_path_list = sorted(xml_path_list)

    for xml_path in xml_path_list:
        directory = xml_path.rsplit('/', 2)[1]
        framename = xml_path.rsplit('/', 2)[2].rsplit('.', 1)[0]
        gg = framename.rsplit('_')
        diire = gg[0]+'_'+gg[1]

        output_dir = os.path.join(args.xml_dir, diire,'supervised')

        # ディレクトリ作成
        if not (os.path.exists(output_dir)):
            os.makedirs(output_dir)
        #args.xml_dir.rsplit('/', 1)[0]
        before_img_path = os.path.join(args.xml_dir, 'image', framename + '.jpg')
        before_xml_path = xml_path

        # 画像をディレクトリに移動
        if os.path.exists(before_xml_path):
            #shutil.move(before_img_path, os.path.join(output_dir, framename + '.jpg'))
            shutil.move(before_xml_path, os.path.join(output_dir, framename + '.xml'))
            #shutil.move(before_img_path, os.path.join(output_dir, framename + '.jpg'))

