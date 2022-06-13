from __future__ import print_function
import torch
import json
import numpy as np
import glob
from torchvision import datasets, transforms
from PIL import Image
c = "goldfish"
class_idx = None
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# load file names, then in utils loop over file names, read images and get label using folder from path and json .. close file after finishing.
image_files = {}
classes_dictionary = json.load(open("./data/imagenet_class_index.json"))
for k, v in classes_dictionary.items():
    label = int(k)
    parent_folder = v[0]
    files = glob.glob("./data/val_imagenet/" + parent_folder + "/*.JPEG")
    for file in files:
        image_files[file] = label
# shuffling images
l = list(image_files.items())
np.random.shuffle(l)
image_files = dict(l)
for image_file, label in image_files.items():
    img = Image.open(image_file)
    try:
        img = preprocess(img)
    except:
        # these images are not 3 channels but rather black and white ( 1 channel ), so ignore them
        continue
