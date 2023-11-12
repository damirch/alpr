import xml.etree.cElementTree as ET

class PlateBoundingBox:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        self.image_name = self.root.find('filename').text
        
        bbox = self.root.find('object').find('bndbox')
        self.xmin = int(bbox.find('xmin').text)
        self.ymin = int(bbox.find('ymin').text)
        self.xmax = int(bbox.find('xmax').text)
        self.ymax = int(bbox.find('ymax').text)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.center = (self.xmin + self.width/2, self.ymin + self.height/2)
        self.area = self.width * self.height

        # find original image size
        size = self.root.find('size')
        self.image_width = int(size.find('width').text)
        self.image_height = int(size.find('height').text)

        self.width01 = self.width / self.image_width
        self.height01 = self.height / self.image_height

        self.center01 = (self.center[0] / self.image_width, self.center[1] / self.image_height)

    def describe(self):
        print("PlateBoundingBox")
        print("xml_path: {}".format(self.xml_path))
        print("image_name: {}".format(self.image_name))
        print("image_width: {}".format(self.image_width))
        print("image_height: {}".format(self.image_height))
        print("xmin: {}".format(self.xmin))
        print("ymin: {}".format(self.ymin))
        print("xmax: {}".format(self.xmax))
        print("ymax: {}".format(self.ymax))
        print("width: {}".format(self.width))
        print("height: {}".format(self.height))
        print("center: {}".format(self.center))
        print("area: {}".format(self.area))
        print("")