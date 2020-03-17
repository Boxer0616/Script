from lxml import etree

class AnnotationIO:
    objects = []
    folder = ''
    filename = ''
    path = ''
    size = {}
    type_ = ''
    value = ''

    def __init__(self, path):
        self.objects = []
        self.polygons = []
        self.load(path)

    def load(self, path):
        doc = etree.parse(path)

        objs = doc.xpath('//object')

        for obj in objs:
            polys = []
            bdbs = obj.xpath('.//bndbox') 
            for bdb in bdbs:
                self.objects.append({
                'xmin' : int(bdb.xpath('.//xmin')[0].text),
                'xmax' : int(bdb.xpath('.//xmax')[0].text),
                'ymin' : int(bdb.xpath('.//ymin')[0].text),
                'ymax' : int(bdb.xpath('.//ymax')[0].text)
                })

            points = obj.xpath('.//polygon/point')
            for p in points:
                x = int(p.xpath('.//x')[0].text)
                y = int(p.xpath('.//y')[0].text)
                polys.append([x,y])
            self.polygons.append(polys)


        self.folder = doc.xpath('//folder')[0].text
        self.filename = doc.xpath('//filename')[0].text
        self.path = doc.xpath('//path')[0].text
        self.size['width'] = int(doc.xpath('//width')[0].text)
        self.size['height'] = int(doc.xpath('//height')[0].text)
        self.size['depth'] = int(doc.xpath('//depth')[0].text)

def bboxes(anno):
    return [(
        obj['xmin'], 
        obj['ymin'],
        obj['xmax'],
        obj['ymax'])\
        for obj in anno.objects
        ]

def polygons(anno):
    return anno.polygons


if __name__ == '__main__':

    data_name = '09JD191D'
    path = './sample/Xml/{}.xml'.format(data_name)
    ann = AnnotationIO(path)
    print(bboxes(ann))


