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
from keras.preprocessing.image import ImageDataGenerator


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


def create_rcnn_face_detection_network():
        inputs = Input(shape=(227, 227, 3), name='input_tensor')
        # First Convolution
        conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu',
                       name='conv1')(inputs)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
        pool1 = BatchNormalization(name = 'batch_norm_1')(pool1)
        
        # Second Convolution
        conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
                       name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(conv2)
        pool2 = BatchNormalization(name = 'batch_norm_2')(pool2)
        
        # Third Convolution
        conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv3')(pool2)
        conv3 = BatchNormalization(name = 'batch_norm_3')(conv3)
        
        # Fourth Convolution
        conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv4')(conv3)
        conv4 = BatchNormalization(name = 'batch_norm_4')(conv4)
        
        # Fifth Convolution
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                       name='conv5')(conv4)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)
        pool5 = BatchNormalization(name = 'batch_norm_5')(pool5)
        
        # Flatten the output of fifth covolution part
        flatten = Flatten(name='flatten')(pool5)

        # Fully connected with 4096 units
        fully_connected = Dense(4096, activation='relu', name='fully_connected')(flatten)

        # Fully connected with 512 units
        face_detection = Dense(512, activation='relu', name='detection')(fully_connected)
        
        

        # Face detection output with 2 units
        face_output = Dense(2, name='face_detection_output')(face_detection)

        model = Model(inputs=inputs, outputs=face_output)

        model.compile(Adam(lr=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

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

def face_rcnn_callbacks():
    path_checkpoint = 'face_rcnn_checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1,
                                          save_weights_only=True, save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, restore_best_weights =True)
    log_dir = 'face_rcnn_logs'
    callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    return [callback_checkpoint, callback_early_stopping, callback_tensorboard]

def initiliaze_weights_alexnet(model,weights_dic):
    model.get_layer('conv1').set_weights([weights_dic['conv1'][0], weights_dic['conv1'][1]])
    conv2_weights = np.append(weights_dic['conv2'][0],weights_dic['conv2'][0],axis=2)
    model.get_layer('conv2').set_weights([conv2_weights, weights_dic['conv2'][1]])
    model.get_layer('conv3').set_weights([weights_dic['conv3'][0], weights_dic['conv3'][1]])
    conv4_weights = np.append(weights_dic['conv4'][0],weights_dic['conv4'][0],axis=2)
    model.get_layer('conv4').set_weights([conv4_weights, weights_dic['conv4'][1]])
    conv5_weights = np.append(weights_dic['conv5'][0],weights_dic['conv5'][0],axis=2)
    model.get_layer('conv5').set_weights([conv5_weights, weights_dic['conv5'][1]])
    return model

#X1 = np.load('helen1029.npy',allow_pickle=True)
#X2 = np.load('helen1233.npy',allow_pickle=True)
#X3 = np.load('helen1439.npy',allow_pickle=True)
#X4 = np.load('helen208.npy',allow_pickle=True)
#X5 = np.load('helen209.npy',allow_pickle=True)
#X6 = np.load('helen416.npy',allow_pickle=True)
#print('yes  half done')
#X7 = np.load('helen421.npy',allow_pickle=True)
#X8 = np.load('helen615.npy',allow_pickle=True)
#X9 = np.load('helen620.npy',allow_pickle=True)
##X10 = np.load('helen836.npy',allow_pickle=True)
#X11 = np.load('test.npy',allow_pickle=True)
#print('complete ')

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory=r"./train/",
    target_size=(227, 227),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42
)

valid_generator = train_datagen.flow_from_directory(
    directory=r"./validation/",
    target_size=(227, 227),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=42
)
test_generator = train_datagen.flow_from_directory(
    directory=r"./test/",
    target_size=(227, 227),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)




model_save_file = 'face_model.h5'
exists = os.path.isfile(model_save_file)
if not exists:
    face_model = create_rcnn_face_detection_network()
    weights_dic = np.load('bvlc_alexnet.npy', encoding='bytes' ,allow_pickle=True).item()
    face_model = initiliaze_weights_alexnet(face_model,weights_dic)
    callbacks = face_rcnn_callbacks()

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    face_model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=10,
                        callbacks=callbacks
    )

    print('Saving Model')

    face_model.save(model_save_file)

    print('Model Saved')

if exists:

    print('MODEL ALREADY EXISTS!!!!!!!!!!!!!!!')
    #labels = [y_Face, y_Landmarks,  y_Pose]

'''outer_epochs = 25
for j in range(outer_epochs):
    if j >= 1:
        print('##########################')
        print('Saving model')
        face_model.save(model_save_file)
    print ('#######################################')
    print('###########################################')
    print("Outer Epoch: ", j)
    print('#######################################')
    print('######################################')
    for i in X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11:
        X_train, y_Face, y_Landmarks,  y_Pose,  = get_data_and_print_shapes(i)
        
        
        
        
        model_save_file = 'face_model.h5'
        exists = os.path.isfile(model_save_file)
        
        if not exists:
          
            face_model = create_rcnn_face_detection_network()
            weights_dic = np.load('bvlc_alexnet.npy', encoding='bytes' ,allow_pickle=True).item()
            face_model = initiliaze_weights_alexnet(face_model,weights_dic)
            callbacks = face_rcnn_callbacks()
            labels = [y_Face, y_Landmarks,  y_Pose]

        
            face_model.fit(x=X_train, y=y_Face, batch_size=32, validation_split = 0.2, epochs=1, callbacks = callbacks)
        
            print('Saving Model')

            face_model.save(model_save_file)
        
            print('Model Saved')
  
        else:
            if j == 0 and i == X1:
                print('Loading Model')
                face_model = load_model(model_save_file)
                print('Model Loaded') 

            callbacks = face_rcnn_callbacks()
            labels = [y_Face, y_Landmarks,  y_Pose]


            print('training_model')
            face_model.fit(x=X_train, y=y_Face, batch_size=32, validation_split = 0.2, epochs=1, callbacks = callbacks)

            #print('Saving Model')

            #face_model.save(model_save_file)

            #print('Model Saved')'''
        
        
        
