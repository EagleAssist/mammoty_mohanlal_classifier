import numpy as np
import cv2
import keras
CATEGORIES = ['mammooty', 'mohanlal']


def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (224, 224))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 224,224, 1)
    return new_arr
#preprocessing the user input
model = keras.models.load_model('cnn4.model')
result = model.predict([image('dataset/mohanlal/mohanlal774.png')])
if result[0][0] == 0:
    prediction = 'mamooty'
elif result[0][0] == 1:
    prediction = 'mohanlal'
else:
    prediction = 'I dont know this guy!'
print(result)
print(prediction)