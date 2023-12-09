"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np


RDD_CLASSES = ['D00',
               'D10',
               'D20',
               'D40',
               'Repair',
               'Block crack']



def parse_xml(xml_file, RDD_claess=RDD_CLASSES):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text

        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)

        mean_x = (xmin + xmax) / 2.0
        mean_y = (ymin + ymax) / 2.0
        box_width = xmax - xmin
        box_height = ymax - ymin

        # normalize
        mean_x /= width
        mean_y /= height
        box_width /= width
        box_height /= height
        
        # name index
        name_indx = RDD_claess.index(name)

        boxes.append([name_indx, mean_x, mean_y, box_width, box_height])

    return boxes




class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_dir, label_dir, S=7, B=2, C=6, transform=None,
    ):
        """
        Input:
            img_dir: where the image folder is
            label_dir: where the label folder is
            S: S * S is the number of grid cells
            B: number of bounding boxes per grid cell
            C: number of classes
            transform: torchvision.transforms
        """
        self.annotations = os.listdir(label_dir)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations[index])
        boxes = parse_xml(label_path)
        # with open(label_path) as f:
        #     for label in f.readlines():
        #         class_label, x, y, width, height = [
        #             float(x) if float(x) != int(float(x)) else int(x)
        #             for x in label.replace("\n", "").split()
        #         ]

        #         boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations[index].replace("xml", "jpg"))
        image = Image.open(img_path)
        # image = image.resize((448, 448))
        # image = np.array(image)
        # image = torch.tensor(image, dtype=torch.float32)
        # image = image.permute(2, 0, 1)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for k, box in enumerate(boxes):
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, (self.C+1):(self.C+5)] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


def test():
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, bboxes):
            for t in self.transforms:
                img, bboxes = t(img), bboxes

            return img, bboxes
        
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

    dataset = VOCDataset(
        "D:\\Datasets\\RDD2022_China_Drone\\China_Drone\\train\\images",
        "D:\\Datasets\\RDD2022_China_Drone\\China_Drone\\train\\annotations\\xmls",
        transform = transform,
    )

    img, label_matrix = dataset[1]
    print(type(img), type(label_matrix))
    print(label_matrix.shape)
    print(label_matrix)
    
    print(img.shape)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    # make unit8
    # img = img.astype(np.uint8)
    
    print(f"min: {np.min(img)}, max: {np.max(img)}")
    plt.imshow(img)
    plt.show()

    
if __name__ == "__main__":
    test()