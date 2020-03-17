import os
import glob
import shutil
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml_dir', default='./output/label')
parser.add_argument('-y', '--out_dir', default='./output/image')
parser.add_argument('-z', '--out_dirdd', default='./output/ff')
args = parser.parse_args()



#ファイル名とディレクトリ名
def path_to_name(data_path):
    #directory = data_path.rsplit('/', 3)[1]
    framename = data_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    ddd3 = framename.split('_')
    directory = ddd3[0] + '_' + ddd3[1]
    #tree = ET.parse(data_path)
    #filename = tree.find('filename').text
    #directory = tree.find('folder').text

    return directory,framename

"""
---------------------------------------------------------------------------------------------------

メイン関数

---------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    xml_path_list = glob.glob(os.path.join(args.xml_dir, '**'))
    out_path_list = glob.glob(os.path.join(args.out_dir, '**'))
    xml_path_list = sorted(xml_path_list)
    out_path_list = sorted(out_path_list)
    path_supervised = 'supervised'

    for xml_path in xml_path_list:
        directory,framename = path_to_name(xml_path)
        output_dir_xml = os.path.join(args.out_dirdd,directory,path_supervised)

        # ディレクトリ作成
        if not (os.path.exists(output_dir_xml)):
            os.makedirs(output_dir_xml)
        # ディレクトリwo移動
        #shutil.copyfile(xml_path, output_dir_xml+'/'+framename+'.xml')
        shutil.move(xml_path, output_dir_xml)

    for out_path in out_path_list:
        directory2,framename2 = path_to_name(out_path)
        output_dir_img = os.path.join(args.out_dirdd,directory2)
        # ディレクトリwo移動
        #shutil.copyfile(out_path, output_dir_img+ '/' + framename2+ '.jpg')
        shutil.move(out_path, output_dir_img)
