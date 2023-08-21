
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
import pickle

dict=r'/home/user/mywork/dl/dataset'
cat=['mohanlal','mammooty']
data=[]
img_size=224
for category in cat:
    folder= os.path.join(dict,category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        label=cat.index(category)
        img_arry=cv.imread(img_path,cv.IMREAD_GRAYSCALE)
        img_arry=cv.resize(img_arry,(img_size,img_size))
        data.append([img_arry,label])
random.shuffle(data)
# to shuffle the data
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)
X = np.array(X)
y = np.array(y)
X = pickle.dump(X,open('X.pkl', 'wb'))
y = pickle.dump(y,open('y.pkl', 'wb'))

# storing the data in a pickle file