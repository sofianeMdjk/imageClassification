from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras.preprocessing.image as img
import numpy as np
from keras.preprocessing import image


classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



train_data = img.ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_data = img.ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory('training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_data.flow_from_directory('test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')


classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)

classifier.save('model.h5')
