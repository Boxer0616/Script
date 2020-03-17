import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml_dir', default='/home/hiroshi/Dataset/20190709/xml')
args = parser.parse_args()


"""
---------------------------------------------------------------------------------------------------

メイン関数

---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    xml_path_list = glob.glob(os.path.join(args.xml_dir, '**', '*.xml'))
    xml_path_list = sorted(xml_path_list)

    for xml_path in xml_path_list:
        directory = xml_path.rsplit('/', 2)[1]
        framename = xml_path.rsplit('/', 2)[2].rsplit('.', 1)[0]

        output_dir = os.path.join(args.xml_dir.rsplit('/', 1)[0], 'image', directory)

        # ディレクトリ作成
        if not (os.path.exists(output_dir)):
            os.makedirs(output_dir)

        before_img_path = os.path.join(args.xml_dir.rsplit('/', 1)[0], 'image', framename + '.jpg')

        # 画像をディレクトリに移動
        if os.path.exists(before_img_path):
            shutil.move(before_img_path, os.path.join(output_dir, framename + '.jpg'))
            #shutil.move(before_img_path, os.path.join(output_dir, framename + '.jpg'))

