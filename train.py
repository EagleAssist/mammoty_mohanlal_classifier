#building the model
#importing pickle file
import pickle
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))
X = X/255
print(X.shape)
X = X.reshape(-1, 224, 224, 1)
print(X.shape)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
model=Sequential()
# Step 1 - Convolution
model.add(Conv2D(64,(3,3), activation='relu'))
# Step 2 - Pooling
model.add(MaxPooling2D((2,2)))

# Adding a second convolutional layer
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full Connection
model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

# Step 5 - Output Layer
model.add(Dense(2, activation= 'sigmoid'))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X, y, epochs=10, validation_split=0.2)
scores = model.evaluate(X,y,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#to know accuracy of model
model.save('cnn4.model',save_format='h5')
#to save the model