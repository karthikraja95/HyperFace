import numpy as np
import scipy
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import os
from IPython.display import SVG
import os
import cv2
import os.path
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, transform
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


def resize_images_to_227_x_227(images, i):
    images = images / 255
    return np.array([scipy.misc.imresize(im, (227,227)) for im in i[:,0]])

def get_data_and_print_shapes(i):
    data = resize_images_to_227_x_227(i[: 800, 0], i)
    print("dataset shape: ", data.shape)
    
    X_train = np.array([d for d in i[:800, 0]])
    print("training data shape", X_train.shape)
    
    X_test = np.array([d for d in i[800: , 0]])
    print("testing data shape", X_test.shape)
    
    #X_test = resize_images_to_227_x_227(i[800 :, 0], i)
    #print("testing data shape: ", X_test.shape)

    # You can change the array in the second element from [1, 1] to [1] and [0, 0] to [0] .
    y_Face = np.array([face[0] for face in i[: 800, 1]])
    
    print("Face label shape: ", y_Face.shape)
    
    y_Face_test = np.array([face[0] for face in i[800:, 1]])
    print("Face label test shape: ", y_Face_test.shape)

    y_Landmarks = np.array([mark for mark in i[: 800, 2]])
    print("Landmarks label shape: ", y_Landmarks.shape)
    
    
    y_Landmarks_test = np.array([mark for mark in i[800 :, 2]])
    print("Landmarks label test shape: ", y_Landmarks_test.shape)



    
    y_Pose = np.array([pose for pose in i[: 800, 3]])
    print("Pose label shape: ", y_Pose.shape)
    
    
    y_Pose_test = np.array([pose for pose in i[800:, 3]])
    print("Pose label shape test: ", y_Pose_test.shape)


   
    
    return X_train, y_Face, y_Landmarks, y_Pose, X_test, y_Face_test, y_Landmarks_test, y_Pose_test


def create_hyperface_network():
    input = Input(shape=(227, 227, 3), name='input')

    # First Convolution
    conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                   name='conv1')(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
    pool1 = BatchNormalization(name = 'batch_norm_1')(pool1)
    
    # Extracting low-level details from poo11 layer to fuse (concatenate) it later
    conv1a = Conv2D(filters=256, kernel_size=(4, 4), strides=(4, 4), activation='relu', name='conv1a')(pool1)
    conv1a = BatchNormalization(name = 'batch_norm_1a')(conv1a)
    
    # Second Convolution
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                   name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2)
    pool2 = BatchNormalization(name = 'batch_norm_2')(pool2)
    
    # Third Convolution
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                   name='conv3')(pool2)
    conv3 = BatchNormalization(name = 'batch_norm_3')(conv3)
    
    # Extracting mid-level details from conv3 layer to fuse (concatenate) it later with high-level pool5 layer.
    conv3a = Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu',
                    name='conv3a')(conv3)
    conv3a = BatchNormalization(name = 'batch_norm_3a')(conv3a)
    
    # Fourth Convolution
    conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                   name='conv4')(conv3)
    conv4 = BatchNormalization(name = 'batch_norm_4')(conv4)
    
    # Fifth Convolution
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                   name='conv5')(conv4)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)
    pool5 = BatchNormalization(name = 'batch_norm_5')(pool5)
    
    # Fuse (concatenate) the conv1a, conv3a, pool5 layers
    concat = Concatenate(axis=-1, name='concat_layer')([conv1a, conv3a, pool5])
    concat = BatchNormalization(name = 'batch_norm_concat')(concat)
    
    # Add convolution to reduce the size of concatenated layers
    conv_all = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv_all')(concat)

    # Flatten the output of concatenated layer with reduced filters to 192
    flatten = Flatten(name='flatten_layer')(conv_all)

    # Fully connected with 3072 units
    fc_full = Dense(3072, activation='relu', name='fully')(flatten)

    # Split the network into five separate branches with 512 units from fc_full
    # corresponding to the different tasks.
    detection = Dense(512, activation='relu', name='face_detection')(fc_full)
    landmarks = Dense(512, activation='relu', name='landmarks')(fc_full)

    visibility = Dense(512, activation='relu', name='visibility')(fc_full)
    
    pose = Dense(512, activation='relu', name='pose')(fc_full)

    gender = Dense(512, activation='relu', name='gender')(fc_full)
  

    # Face detection output with 2 units
    face_output = Dense(1, name='face_detection_out', activation='softmax')(detection)
    # Landmark localization output with 42 units
    landmarks_output = Dense(42, name='landmarks_output')(landmarks)

    visibility_output = Dense(21, name='visibility_output')(visibility)
    
    # Pose output with 3 units
    pose_output = Dense(3, name='pose_output', activation='softmax')(pose)

    gender_output = Dense(1, name='gender_output', activation='softmax')(gender)
    
    model = Model(inputs=input, outputs=[face_output, landmarks_output, visibility_output,  pose_output, gender_output])
    
    # These losses will be added in keras at the time of optimization
    loss = {'face_detection_out': binary_crossentropy, 'landmarks_output': mean_squared_error, 'visibility_output' : mean_squared_error,
             'pose_output': mean_squared_error, 'gender_output' : binary_crossentropy}
    
    loss_weights = {'face_detection_out': 1, 'landmarks_output': 5.0, 'visibility_output': 0.5, 'pose_output': 5.0, 'gender_output': 2.0}
    model.compile(Adam(lr=0.0001), loss = loss , loss_weights = loss_weights, metrics=['accuracy'])
    return model


def initialize_weights_of_hyperface_with_face_detection_layer_weights(alex_layers):
    for layer in alex_layers:
        #print(layer)
        hyperface_model.get_layer(layer).set_weights(face_model.get_layer(layer).get_weights())
    #print('done')


def return_common_layers_name(hyperface_model, face_detection_model):
    return {layer.name for layer in hyperface_model.layers}.intersection({layer.name for layer in face_detection_model.layers})


def hyperface_callbacks():
    path_checkpoint = 'hyperface_checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_landmarks_output_loss', verbose=1,
                                          save_weights_only=True, save_best_only=True)

    #callback_early_stopping = EarlyStopping(monitor='val_landmarks_output_loss', patience=3, verbose=1)
    log_dir = 'hyperface_logs'
    callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    #return [callback_checkpoint, callback_early_stopping, callback_tensorboard]
    return [callback_checkpoint, callback_tensorboard]


def generator(x, y, batch_size=32):
    data_size = x.shape[0]
    while True:
        batch_start = 0
        while batch_start < data_size:
            batch_x = x[batch_start:batch_start + batch_size, :, :]
            batch_f = y[0][batch_start:batch_start + batch_size]
            batch_l = y[1][batch_start:batch_start + batch_size, :]
            batch_v = y[2][batch_start:batch_start + batch_size, :]
            batch_p = y[3][batch_start:batch_start + batch_size, :]
            batch_g = y[4][batch_start:batch_start + batch_size, :]
            
            batch_start += batch_size
            yield batch_x, [batch_f, batch_l, batch_v, batch_p, batch_g]





print('Face Model loading')
face_model = load_model('face_model.h5')
print('Face Model Loaded')
gender = np.load('gender.npy', allow_pickle=True)
visibility = np.load('visibility.npy', allow_pickle=True)
landmarks = np.load('landmarks.npy', allow_pickle=True)
pose = np.load('pose.npy', allow_pickle=True)
images = np.load('images.npy',allow_pickle=True)
face = np.load('faces.npy', allow_pickle=True)


x_train, x_test, y_train_landmarks, y_test_landmarks, y_train_visibility, y_test_visibility, y_train_pose, y_test_pose, \
y_train_gender, y_test_gender,y_train_face,y_test_face = train_test_split(images, landmarks, visibility, pose, gender, face,test_size = 0.20, shuffle = True)

x_train = x_train.reshape((x_train.shape[0], 227, 227, 3))
x_test = x_test.reshape((x_test.shape[0], 227, 227, 3))

#x_train = x_train/255
#x_test = x_test/255
      

model_save_file = 'hyperface_model.h5'
exists = os.path.isfile(model_save_file)     

'''
        
if not exists:
    hyperface_model = create_hyperface_network()
    callbacks = hyperface_callbacks()
    labels = [y_train_face, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender]
    labels_test = [y_test_face, y_test_landmarks,y_test_visibility, y_test_pose, y_test_gender]
    
    initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))


    hyperface_model.fit_generator(generator(x_train, labels), validation_data = generator(x_test, labels_test), steps_per_epoch = int(len(x_train)//32) , validation_steps = int(len(x_test)//32),  epochs=30, callbacks = callbacks)

    print('Saving Model')

    hyperface_model.save(model_save_file)

    print('Model Saved')

'''

if exists:

    hyperface_model = create_hyperface_network()
    hyperface_model = load_model('hyperface_model.h5') 
    
    callbacks = hyperface_callbacks()
    labels = [y_train_face, y_train_landmarks, y_train_visibility,  y_train_pose, y_train_gender]
    labels_test = [y_test_face, y_test_landmarks, y_test_visibility, y_test_pose, y_test_gender]
    
    #initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))


    #hyperface_model.fit_generator(generator(x_train, labels), validation_data = generator(x_test, labels_test), steps_per_epoch = int(len(x_train)//32),validation_steps = int(len(x_test)//32),  epochs=30, callbacks = callbacks)
    

    hyperface_model.fit(x=x_train, y=labels, batch_size=32, validation_data=(x_test, labels_test), epochs=190, callbacks = callbacks)



    print('Saving Model')

    hyperface_model.save(model_save_file)

    print('Model Saved')

else:

    hyperface_model = create_hyperface_network()
    #
    callbacks = hyperface_callbacks()
    labels = [y_train_face, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender]
    labels_test = [y_test_face, y_test_landmarks,y_test_visibility, y_test_pose, y_test_gender]
    
    initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))


    #hyperface_model.fit_generator(generator(x_train, labels), validation_data = generator(x_test, labels_test), steps_per_epoch = int(len(x_train)//32) , validation_steps = int(len(x_test)//32),  epochs=30, callbacks = callbacks)
    
    hyperface_model.fit(x=x_train, y=labels, batch_size=32, validation_data=(x_test, labels_test), epochs=190, callbacks = callbacks)

    print('Saving Model')

    hyperface_model.save(model_save_file)

    print('Model Saved')




#X_train, y_Face, y_Landmarks,  y_Pose, X_test, y_Face_test, y_Landmarks_test, y_Pose_test = get_data_and_print_shapes(i)   
    #model_save_file = 'hyperface_model.h5'
    #exists = os.path.isfile(model_save_file)
        
'''if not exists:
  
    hyperface_model = create_hyperface_network()
    callbacks = hyperface_callbacks()
    labels = [y_Face, y_Landmarks, y_Pose]
    labels_test = [y_Face_test, y_Landmarks_test, y_Pose_test]
    
    initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))


    hyperface_model.fit_generator(generator(X_train, labels), validation_data = generator(X_test, labels_test), steps_per_epoch = int(len(X_train)//32) , validation_steps = int(len(X_test)//32),  epochs=1, callbacks = callbacks)

    print('Saving Model')

    hyperface_model.save(model_save_file)

    print('Model Saved')
    '''

'''

if j == 0 and i ==X1:
                    print('Loading Model')
                    hyperface_model = load_model(model_save_file)
                    '''

#print('Loading Model')
#hyperface_model = load_model(model_save_file)
#print('Model Loaded')       
        
        