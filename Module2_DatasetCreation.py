import os
import numpy as np
from nibabel.testing import data_path
import cv2
from os import walk
import matplotlib.pyplot as plt
#import AugmentImages as imgAugment
import nibabel as nib
import pandas as pd
import re

folder = 'dataset'
outFolder = 'nifti-results/'
filenames = next(walk(folder), (None, None, []))[2]

class_data = pd.read_csv('class_info.csv', header=None)
subjects = class_data[0]
classes = class_data[1]

rows = 256
cols = 256

def findNonZero(img) :
    img = np.asarray(img)
    return np.count_nonzero(img)

for count in range(0, len(filenames)) :
    try :
        filename = filenames[count]
        temp = re.findall(r'\d+', filename)
        res = list(map(int, temp))
        if(len(res) != 1) :
            continue
        res = res[0]
        x=subjects[subjects == res]
        idx = x.index[0]
        class_val = classes[idx]
        
        img = nib.load(folder + "/" + filename)
        print(img.shape)
        data = img.get_fdata()
        
        for count2 in range(0, len(data[0][0])) :
            img = data[:,:, count2]
            tot = len(img) * len(img[0])
            try :
                img = img[:,:,0:3]
                tot = len(img) * len(img[0]) * len(img[0][0])
            except :
                print('Dimensions are ok')
                
            nonZero = findNonZero(img)
            ratio = nonZero / tot
            if(ratio < 0.4) :
                continue
            
            fname = outFolder + str(class_val) + "/" + filename + "." + str(count2) + ".png"
            cv2.imwrite(fname, cv2.resize(img, (rows, cols)) )
                
            print('Writing %s' % (fname))
    except :
        print('Continue...')