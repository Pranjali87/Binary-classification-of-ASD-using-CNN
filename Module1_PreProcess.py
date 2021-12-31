import os
from os import walk
import preprocessNII as pni

dataFolder = 'kki'
y=[x[1] for x in os.walk(dataFolder)]
folders = y[0]

for count in range(1, len(folders)) :
    folderName = folders[count]
    fileName = "dataset/" + folderName + ".nii.gz"
    fileName2 = "dataset/" + folderName + "anat.nii.gz"
    
    print('Processing %s' % (folderName))
    pni.performPreprocess(dataFolder + "/" + folderName + "/session_1", fileName)
    pni.performPreprocessAnat(dataFolder + "/" + folderName + "/session_1", fileName2)
    
    
    
    