import xml.etree.ElementTree as ET
import pprint
import argparse
import os
import glob
from collections import Counter
import xml.dom.minidom as md

parser = argparse.ArgumentParser()
parser.add_argument('-ff', '--test', type=str, default='./dataset/mask07')

args = parser.parse_args()

#xmlファイル読み込み
def xmlopen():
        #xml_list = glob.glob(os.path.join(args.test, 'supervised', '*.xml'))
        xml_list = glob.glob(os.path.join(args.test,'label' ,'*.xml'))
        #print(xml_list)
        
        return xml_list

#value検索
def valuesearch(xml_list):
        for xml_path in xml_list:
                tree = ET.parse(xml_path)
                #root
                b_root = tree.getroot()
                #porygon削除、categoryのtype,language削除
                create_xml(xml_path,b_root)
                        
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
"""
--------------------------------------------------------------------------------------------

xmlファイルを作成

--------------------------------------------------------------------------------------------
"""
def create_xml(xml_path, befor_root):
    root = ET.Element('annotation')
    # folderを追加する 
    folder = ET.SubElement(root, 'folder') 
    filename = ET.SubElement(root, 'filename')
    # pathを追加する 
    path = ET.SubElement(root, 'path')
    # sourceを追加する 
    soource = ET.SubElement(root, 'source')
    database = ET.SubElement(soource, 'database')
    # sizeを追加する 
    size = ET.SubElement(root, 'size') 
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    # segmentedを追加する 
    segmented = ET.SubElement(root, 'segmented')

    folder.text = befor_root.find('folder').text
    filename.text = befor_root.find('filename').text
    path.text = xml_path
    database.text = befor_root.find('.//source/database').text
    width.text = befor_root.find('.//size/width').text
    height.text = befor_root.find('.//size/height').text
    depth.text = befor_root.find('.//size/depth').text
    segmented.text = befor_root.find('segmented').text

    for d in befor_root.findall('.//object'):
        # object 
        obj = ET.SubElement(root, 'object')
        pos = ET.SubElement(obj, 'pos')
        pos.text = d.find('pos').text
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = d.find('truncated').text
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = d.find('difficult').text
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = d.find('.//bndbox/xmin').text
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = d.find('.//bndbox/ymin').text
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = d.find('.//bndbox/xmax').text
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = d.find('.//bndbox/ymax').text
        #polygon
        polygon = ET.SubElement(obj, 'polygon')
        polygon_num = ET.SubElement(polygon, 'num')
        polygon_num.text = d.find('.//polygon/num').text
        point_b = d.findall('.//polygon/point')
        for poin in point_b:
                point_a = ET.SubElement(polygon, 'point')
                point_x = ET.SubElement(point_a, 'x')
                point_x.text = poin.find('x').text
                point_y = ET.SubElement(point_a, 'y')
                point_y.text = poin.find('y').text   
        # category
        category = ET.SubElement(obj, 'category')
        typee = ET.SubElement(category, 'type')
        typee.text = d.find('.//category/type').text
        value = ET.SubElement(category, 'value')
        value.text = d.find('.//category/value').text
        language = ET.SubElement(category, 'language')
        language.text = d.find('.//category/language').text



    #tree = ET.ElementTree(root)
    #tree.write(xml_path)
    indent(root)
    #writeXml(root,xml_path)

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
def copy():
      print('')  

if __name__ == '__main__':
        xml_list = xmlopen()
        valuesearch(xml_list)
