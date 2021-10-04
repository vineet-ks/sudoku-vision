import os
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split

X = []
y = []
data_path = "./data"

for number in os.listdir(data_path):
    for img_name in os.listdir(os.path.join(data_path, number)):
        y.append(int(number))
        img = cv2.imread(os.path.join(data_path, number, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        X.append(img)

X = np.asarray(X)
X = X/255
X = X.reshape(X.shape[0], 28, 28, 1)
y = np.asarray(y)
y = np_utils.to_categorical(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
num_classes = 10

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
print(model.summary())

batch_size = 32
epochs = 20

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

print("\n\nTest metrics")
print(model.metrics_names)
print(model.evaluate(X_test, y_test))

model.save("./model", overwrite=True)




