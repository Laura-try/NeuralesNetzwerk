 import lvreader as lv
import numpy as np
import skimage.transform as sk
from pprint import pprint
import os


#creates array of labels
def addLabel ( file, label):
    fileList1 = []

    for f in os.listdir(file):
        if f.endswith('.im7'):
            fileList1.append(f)

    count = len(fileList1)
    start = int(file[-1]) 
    
    for index in range(0, count):
        digit = (start + index) % 10
        label.append(digit)
   
#get the image and refittest into a smaller array
def prepareImage (filepath):
    buffer  = lv.read_buffer(filepath)
    imageOriginal = buffer.as_masked_array()
    imageSmall = sk.rescale(imageOriginal, 0.25, anti_aliasing= False)
    image1D  = imageSmall.flatten()
    return np.rint(image1D)

# creates an array of the immage data
def addImages (file, data):
    fileList2 = []

    for f in os.listdir(file):
        if (f.endswith('.im7')):
             path = "{0}\{1}".format(file, f)
             fileList2.append(path)

    for im in fileList2:
        data.append(prepareImage(im))


label = [0]
data = [0]
root = "F:\RawData\Project_Imaging_210604_134553"
fileList = []

for f in os.listdir(root):
    if not (f.endswith('.set') or f.startswith('P')):
        path = "{0}\{1}".format(root, f)
        fileList.append(path)

for file in fileList:
    addLabel(file, label)
    addImages(file, data)

 #remove first entry was placeholder
label.pop(0)
data.pop(0)

#convert to np array
labelNp = np.array(label, dtype= int)
dataNp = np.array(data, dtype= int)
dataSet = np.column_stack ((labelNp, dataNp))
print(dataSet.shape)
np.save("G:\data025", dataSet)

print(labelNp.shape)
print(dataNp.shape)
