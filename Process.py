from PIL import Image
from math import sqrt
import numpy as np
from pathlib import Path


def get_image_data(f, size = (500, 500)):
    return np.reshape(list(Image.open(f).convert(mode='RGB').resize(size).getdata()), (size[0],size[1],3))

def get_images_from_dir(d, size = (500,500), filetypes = ['.jpg','.png','.jpeg']):
    p = Path(d)
    images = []
    for child in p.iterdir():
        for suff in child.suffixes:
            if suff in filetypes:
                images.append(get_image_data(child))
                break
    return images

def get_data_for_classes(classes, size = (500,500), root_dir = './'):
    labels = []
    data = []
    for cls in classes:
        cls_data = get_images_from_dir(root_dir + cls)
        data.extend(cls_data)
        labels.extend([cls for _ in range(len(cls_data))])
    
    return data, labels

def one_hot(cls, n_classes):
    ret = np.zeros(n_classes)
    ret[cls] = 1
    return ret
