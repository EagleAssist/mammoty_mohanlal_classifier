#building the model
#importing pickle file
import pickle
import keras
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, GlobalAvgPool2D
from keras.models import Model
from keras.optimizers import Adam
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))
X = X/255
print(X.shape)
# X = X.reshape(224, 224, 3)
print(X.shape)
num_classes = 2  # Replace with the actual number of classes in your dataset

cnn_base=VGG19(weights='imagenet',include_top=False,input_shape=(224, 224, 3))
print('vgg19 loaded')
print(cnn_base.summary())
# Freeze the pre-trained layers to avoid updating their weights
for layer in cnn_base.layers:
    layer.trainable = False
# Add custom classification layers on top of VGG19
x = cnn_base.output
x = GlobalAvgPool2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=cnn_base.input, outputs=predictions)

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

def load_and_preprocess_data(dataset):
    data = []
    labels = []
    for class_name in os.listdir(dataset):
        class_dir = os.path.join(dataset, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = img / 255.0  # Rescale pixel values to [0, 1]
            data.append(img)
            labels.append(class_name)  # Assuming the class folder name is the label
    return data, labels
train_data, train_labels = load_and_preprocess_data('dataset')
#validation_data, validation_labels = load_and_preprocess_data('validation')

#train_labels = keras.utils.to_categorical(train_labels, 2)
encoder = LabelEncoder()
encoder.fit(train_labels)
label = encoder.transform(train_labels)
# y_test = encoder.transform(y_test)
train_label = keras.utils.to_categorical(label, 2)
# print(train_label)
# Train the model
model.fit(
    tf.convert_to_tensor(train_data),
    train_label,
    
    epochs=10,
)
# Save the trained model
model.save('fine_tuned_vgg19.h5')





# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
# model=Sequential()
# # Step 1 - Convolution
# model.add(Conv2D(64,(3,3), activation='relu'))
# # Step 2 - Pooling
# model.add(MaxPooling2D((2,2)))

# # Adding a second convolutional layer
# model.add(Conv2D(64,(3,3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

# # Step 3 - Flattening
# model.add(Flatten())

# # Step 4 - Full Connection
# model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

# # Step 5 - Output Layer
# model.add(Dense(2, activation= 'sigmoid'))

# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
# model.fit(X, y, epochs=10, validation_split=0.2)
# scores = model.evaluate(X,y,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# #to know accuracy of model
# model.save('cnn4.model',save_format='h5')
# #to save the model