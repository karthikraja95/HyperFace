import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
#import selectivesearch
import matplotlib.patches as mpatches
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy, mean_squared_error
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import os
from IPython.display import SVG

def plot_images(images, face_label = None, gender_label = None):
    fig, axes = plt.subplots(3, 6 , figsize=(20,10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if face_label is not None:
            face = 'Face' if face_label[i][0] else 'No Face'
            shape = 'Shape: ' + str(images[i].shape)
            xlabel = face +'\n' + shape
        else:
            shape = 'Shape: ' + str(images[i].shape)
            xlabel = shape
        ax.set_xlabel(xlabel)        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def resize_images_to_227_x_227(images, i):
    images = images / 255
    return np.array([scipy.misc.imresize(im, (227,227)) for im in i[:,0]])

def get_data_and_print_shapes(i):
    X_train = resize_images_to_227_x_227(i[:, 0], i)
    print("training data shape: ", X_train.shape)

    # You can change the array in the second element from [1, 1] to [1] and [0, 0] to [0] .
    y_Face = np.array([face[0] for face in i[:, 1]])
    
    print("Face label shape: ", y_Face.shape)

    y_Landmarks = np.array([mark for mark in i[:, 2]])
    print("Landmarks label shape: ", y_Landmarks.shape)


    
    y_Pose = np.array([pose for pose in i[:, 3]])
    print("Pose label shape: ", y_Pose.shape)

   
    
    return X_train, y_Face, y_Landmarks, y_Pose


def selective_search(img, plot_image = False):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, sigma=0.9, min_size=100)
    candidates = set()
    for r in regions:
        x, y, w, h = r['rect']
        candidates.add(r['rect'])
    if plot_image:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        for x, y, w, h in candidates:
            rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
        plt.show()
    return candidates

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_xy(box):
    x,y,w,h = box
    x_top_left = x
    y_top_left = y
    x_right_bottom = x_top_left + w
    y_right_bottom = y_top_left + h
    return [x_top_left, y_top_left, x_right_bottom, y_right_bottom]


def region_of_proposals(image):
    candidates = selective_search(image, True)   
    selected_regions = list()
    boxA = [0, 0,image.shape[0], image.shape[1]]
    for candidate in candidates:
        boxB = get_xy(candidate)
        iou = intersection_over_union(boxA, boxB)
        if iou >= 0.5:
            selected_regions.append(candidate)
    # resize according to alexnet input and return selected regions
    image = image / 255
    return np.array([scipy.misc.imresize(image[r[0]:r[3], r[1]:r[2],:], (227, 227)) for r in selected_regions])

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
    
    pose = Dense(512, activation='relu', name='pose')(fc_full)
  

    # Face detection output with 2 units
    face_output = Dense(2, name='face_detection_out', activation='softmax')(detection)
    # Landmark localization output with 42 units
    landmarks_output = Dense(136, name='landmarks_output')(landmarks)
    
    # Pose output with 3 units
    pose_output = Dense(3, name='pose_output', activation='softmax')(pose)
    
    model = Model(inputs=input, outputs=[face_output, landmarks_output,  pose_output])
    
    # These losses will be added in keras at the time of optimization
    loss = {'face_detection_out': sparse_categorical_crossentropy, 'landmarks_output': mean_squared_error,
             'pose_output': mean_squared_error}
    
    loss_weights = {'face_detection_out': 1, 'landmarks_output': 5,  'pose_output': 5}
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
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1,
                                          save_weights_only=True, save_best_only=True)

    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    log_dir = 'hyperface_logs'
    callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    return [callback_checkpoint, callback_early_stopping, callback_tensorboard]


def generator(x, y, batch_size=32):
    data_size = x.shape[0]
    while True:
        batch_start = 0
        while batch_start < data_size:
            batch_x = x[batch_start:batch_start + batch_size, :, :]
            batch_f = y[0][batch_start:batch_start + batch_size]
            batch_l = y[1][batch_start:batch_start + batch_size, :]
           
            batch_p = y[2][batch_start:batch_start + batch_size, :]
            
            batch_start += batch_size
            yield batch_x, [batch_f, batch_l,  batch_p]

def predict_and_save_image(j):
    image = cv2.imread('face.jpg')
    image = cv2.resize(image,(227,227))
    prediction = hyperface_model.predict(image.reshape(1,227,227,3))
    for i in range(0,68):
      x_cord = int(prediction[1][0][i+68]*227)
      y_cord = int(prediction[1][0][i]*227)
      cv2.circle(image,(x_cord,y_cord),1,(255,0,0),-1)
    cv2.imwrite('./images/'+str(j)+'.jpg',image)
    print('################ IMAGE SAVED \###################')

print('Face Model loading')
face_model = load_model('face_model.h5')
print('Face Model Loaded')

X1 = np.load('helen1029.npy',allow_pickle=True)
X2 = np.load('helen1233.npy',allow_pickle=True)
X3 = np.load('helen1439.npy',allow_pickle=True)
X4 = np.load('helen208.npy',allow_pickle=True)
X5 = np.load('helen209.npy',allow_pickle=True)
X6 = np.load('helen416.npy',allow_pickle=True)
print('Half Dataset Loaded')
X7 = np.load('helen421.npy',allow_pickle=True)
X8 = np.load('helen615.npy',allow_pickle=True)
X9 = np.load('helen620.npy',allow_pickle=True)
X10 = np.load('helen836.npy',allow_pickle=True)
X11 = np.load('test.npy',allow_pickle=True)
print('Complete Dataset Loaded')




        
outer_epochs = 25

             
for j in range(outer_epochs):
    if j >= 1:
        print('##########################')
        print('Saving model')
        hyperface_model.save(model_save_file)
        predict_and_save_image(j)
    print ('#######################################')
    print('###########################################')
    print("Outer Epoch: ", j)
    print('#######################################')
    print('######################################')
    for i in X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11:
      
        

        X_train, y_Face, y_Landmarks,  y_Pose,  = get_data_and_print_shapes(i)
        
        
       
        model_save_file = 'hyperface_model.h5'
        exists = os.path.isfile(model_save_file)
        
        if not exists:
          
            hyperface_model = create_hyperface_network()
            callbacks = hyperface_callbacks()
            labels = [y_Face, y_Landmarks, y_Pose]
            
            initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))

        
            hyperface_model.fit_generator(generator(X_train, labels), steps_per_epoch = 32 , epochs=1, callbacks = callbacks)
        
            print('Saving Model')

            hyperface_model.save(model_save_file)
        
            print('Model Saved')
  
        else: 
          
            if j == 0 and i == X1:
                print('Loading Model')
                hyperface_model = load_model(model_save_file)
                print('Model Loaded')

                #hyperface_model = create_hyperface_network()
            print('loading callbacks')
            callbacks = hyperface_callbacks()
            print('loading lables')
            labels = [y_Face, y_Landmarks,  y_Pose]
            
            #initialize_weights_of_hyperface_with_face_detection_layer_weights(return_common_layers_name(face_model,hyperface_model))

        
            print('training model')
            hyperface_model.fit_generator(generator(X_train, labels), steps_per_epoch = 32, epochs=1, callbacks = callbacks)
            print('training complete')
        
            #print('Saving Model')

            #hyperface_model.save(model_save_file)

            #print('Model Saved')
        
        
        
