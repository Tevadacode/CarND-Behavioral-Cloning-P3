import os
import csv

# read each line from driving_log.csv to samples
samples = []
with open('./3_laps/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:len(samples)]
   
import cv2
import random
import numpy as np
import sklearn
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
# split data set for training and validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# use python generator to fit data in the memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './3_laps/IMG/' + batch_sample[0].split('\\')[-1]
                left_name = './3_laps/IMG/' + batch_sample[1].split('\\')[-1]
                right_name = './3_laps/IMG/' + batch_sample[2].split('\\')[-1]
                center_image = mpimg.imread(name)
                left_image = mpimg.imread(left_name)
                right_image = mpimg.imread(right_name)

                correction = 0.5
                center_angle = float(batch_sample[3])
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(steering_left)
                angles.append(steering_right)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Input
from keras.layers import Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.backend import tf as ktf
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
# Data normalization
model.add(Lambda(lambda x: ktf.image.resize_images(x, (80, 160)) , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((25,12),(0,0))))
model.add(Lambda(lambda x: (x/127.5) - 1.0))

# convnet architecture adapting NVIDIA architecture
model.add(Conv2D(24,(5,5), strides=(2,2), kernel_initializer='truncated_normal', activation="relu"))
model.add(Dropout(0.5))          
model.add(Conv2D(36, (5,5), kernel_initializer='truncated_normal', activation="relu"))
model.add(Dropout(0.5))
model.add(Conv2D(48, (5,5), kernel_initializer='truncated_normal', activation="relu"))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), kernel_initializer='truncated_normal', activation="relu"))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), kernel_initializer='truncated_normal', activation="relu"))
model.add(Dropout(0.5))
          
model.add(Flatten())
          
model.add(Dense(100, kernel_initializer='truncated_normal'))
model.add(Dense(50, kernel_initializer='truncated_normal'))
model.add(Dense(1))

# configure the model using mean square error and Adam optimizer
model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks = [early_stopping, checkpoint]
# train the model using fit_generator
model.fit_generator(train_generator, steps_per_epoch = 4*len(train_samples)/32,\
                    validation_data=validation_generator,\
                    validation_steps=4*len(validation_samples)/32, epochs=150,\
                    verbose=2, callbacks = callbacks)
                    
model.save_weights('model.h5')
print ('Model is saved')
