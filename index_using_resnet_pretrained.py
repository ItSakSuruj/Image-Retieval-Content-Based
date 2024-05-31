# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:09:55 2024

@author: SURUJ_KALITA
"""

import os
import h5py
import numpy as np
from Resnet_feature_extractor import getResNet50Model


images_path ="img/"
img_list = [os.path.join(images_path,f) for f in os.listdir(images_path)]


print("  start feature extraction ")


model = getResNet50Model()

path = "img/"

feats = []
names = []

for im in os.listdir(path):  #iterate through all images to extract features
    print("Extracting features from image - ", im)
    X = model.extract_feat(path+im)

    feats.append(X)
    names.append(im)
    
feats = np.array(feats)

# directory for storing extracted features
output = "ResnetFeatures.h5"

print(" writing feature extraction results to h5 file")


h5f = h5py.File(output, 'w')
h5f.create_dataset('dataset_1', data = feats)
h5f.create_dataset('dataset_2', data = np.string_(names))
h5f.close()