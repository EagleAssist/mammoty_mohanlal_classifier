

import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
correct_predictions=0
total_predictions =0

# Load the trained VGG19 model
model = load_model('fine_tuned_vgg19.h5')

# Directory where the test images are stored
test_data_dir = 'dataset'  # Change this to the directory containing your test images

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0  # Rescale pixel values to [0, 1]
    return img

# Make predictions for each image in the test data directory
for class_name in os.listdir('dataset'):
    class_dir = os.path.join(test_data_dir, class_name)
    
    for filename in os.listdir(class_dir):
        

        img_path = os.path.join(class_dir, filename)
        image = load_and_preprocess_image(img_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)  # Get the class with the highest probability
        class_labels = ['mammooty', 'mohanlal']  # Replace with your actual class labels
        predicted_class_label = class_labels[predicted_class]
        print(f"Image: {filename}, Predicted Class: {predicted_class_label}")
        if filename.startswith(class_labels[0]) and predicted_class_label==class_labels[0]:
            correct_predictions += 1
        elif filename.startswith(class_labels[1]) and predicted_class_label==class_labels[1]:
            correct_predictions += 1
        total_predictions += 1

        
        

accuracy = (correct_predictions / total_predictions) * 100.0
print(f"Accuracy: {accuracy:.2f}%")
# You can replace 'class_1' and 'class_2' in the class_labels list with your actual class labels.

# import numpy as np
# import cv2
# import keras
# CATEGORIES = ['mammooty', 'mohanlal']


# def image(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     new_arr = cv2.resize(img, (224, 224))
#     new_arr = np.array(new_arr)
#     new_arr = new_arr.reshape(-1,224, 224, 1)
#     return new_arr
    
# #preprocessing the user input
# model = keras.models.load_model('fine_tuned_vgg19.h5')
# result = model.predict([image('dataset/mohanlal/mohanlal774.png')])
# if result[0][0] == 0:
#     prediction = 'mamooty'
# elif result[0][0] == 1:
#     prediction = 'mohanlal'
# else:
#     prediction = 'I dont know this guy!'
# print(result)
# print(prediction)