import xml.etree.cElementTree as ET

class PlateBoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, image_width, image_height):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
 
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.center = (xmin + self.width/2, ymin + self.height/2)
        self.area = self.width * self.height
 
        self.width01 = self.width / image_width
        self.height01 = self.height / image_height

        self.center01 = (self.center[0] / image_width, self.center[1] / image_height)

    @staticmethod
    def load_from_xml(xml_path: str):
        xml_path = xml_path
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_name = root.find('filename').text
        
        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
        
            bboxes.append((xmin, ymin, xmax, ymax))

        # find original image size
        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)

        return [PlateBoundingBox(xmin, ymin, xmax, ymax, image_width, image_height) for xmin, ymin, xmax, ymax in bboxes]

    def describe(self):
        print("PlateBoundingBox")
        print("xmin: {}".format(self.xmin))
        print("ymin: {}".format(self.ymin))
        print("xmax: {}".format(self.xmax))
        print("ymax: {}".format(self.ymax))
        print("width: {}".format(self.width))
        print("height: {}".format(self.height))
        print("width01: {}".format(self.width01))
        print("height01: {}".format(self.height01))
        print("center01: {}".format(self.center01))
        print("center: {}".format(self.center))
        print("area: {}".format(self.area))
        print("")