
import lvreader as lv
import numpy as np
import skimage.transform as sk
from pprint import pprint
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes.ravel()
# get im7 
buffer  = lv.read_buffer("G:\RawData\Project_Imaging_210604_134553\A-0\B00002.im7")


#change attributes of image works like dict
buffer.attributes["CustomImageTag_Count"] = 1
pprint(buffer.attributes)

#buffer as array
arr = buffer.as_masked_array()
b = arr.flatten()
print(arr.shape)

print(b.shape)
c = sk.rescale(arr, 0.25, anti_aliasing= False)
print(c.shape)

bufferTest = lv.read_buffer("G:\\testSmall.im7")
ax[0].imshow(arr, cmap='gray')
ax[1].imshow(c, cmap='gray')
plt.tight_layout()
plt.show()
