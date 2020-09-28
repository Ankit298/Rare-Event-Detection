# Code for the Resnet 50 model is built on top of the starter code from Andrew Ng's Deep Learning Specialization course 

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
import pandas as pd 
import matplotlib.image as img 
import time
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support

train = np.zeros((3828,64,64,1))
test = np.zeros((667,64,64,1))


metadata = pd.read_csv('/content/lesion_final/HAM10000_metadata.csv')
image_number = []
for string in metadata['image_id']:
    image_number.append(int(string[7:13]))
image_number = np.array(image_number)
metadata['image_number'] = image_number
metadata = metadata.sort_values(by=['image_number'])


df_dataframe = metadata.loc[metadata['dx'] == 'df']


metadata_withoutdf = metadata[metadata.dx != 'df']

train_metadata,test_metadata = np.split(metadata_withoutdf, [int(.75*len(metadata))])
train_metadata = train_metadata[0:1914]
test_metadata = test_metadata[0:638]

train_df, test_df = np.split(df_dataframe, [int(.75*len(df_dataframe))])


test_metadata = test_metadata.append(test_df)
test_metadata = test_metadata.sample(frac=1)

#Oversampling
oversamp_factor = 22
for i in range(oversamp_factor):
  train_metadata = train_metadata.append(train_df)
train_metadata = train_metadata.append(train_df.iloc[0:22])
train_metadata = train_metadata.sample(frac=1)

start_time = time.time()
img_size   = (64,64)
itrain = -1
itest = -1
for isic in train_metadata['image_id']:
    if int(isic[7:13]) > 31429:
        image = img.imread("/content/lesion_final/"+"GreyScale"+isic+".jpg")
    elif int(isic[7:13]) <= 31429:
        image = img.imread("/content/lesion_final/"+isic+".jpg")
    image = resize(image,img_size)
    image_array = np.array(image)
    image_array = image_array/255
    itrain += 1
    train[itrain,:,:,0] = image_array
    

for isic_test in test_metadata['image_id']:
    image = img.imread("/content/lesion_final/"+"GreyScale"+isic_test+".jpg")
    image = resize(image,img_size)
    image_array = np.array(image)
    image_array = image_array/255
    itest += 1
    test[itest,:,:,0] = image_array
    

print('Load time: {0} seconds'.format(time.time() - start_time))

label_list = []
label_list_test = []

for label in train_metadata['dx']:
    label_list.append(label)
for label_test in test_metadata['dx']:
    label_list_test.append(label_test)


i = 0
j = 0
list1 = []
list2 = []
for lesion in label_list:
    if lesion == 'df':
      list1.append(1)
      i += 1
    else:
      list1.append(0)
      i += 1

for lesion in label_list_test:
    if lesion == 'df':
      list2.append(1)
      j += 1
    else:
      list2.append(0)
      j += 1

X_train = train
X_test = test
y_train = np.array(list1)
encoder = OneHotEncoder(sparse = False)
y_train = encoder.fit_transform(y_train.reshape(len(y_train),1))
y_test = np.array(list2)
y_test = encoder.fit_transform(y_test.reshape(len(y_test),1))

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path 
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    #SHORTCUT PATH 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut =BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    return X

def ResNet50(input_shape = (64, 64, 1), classes = 2):
    
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='d', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='f')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='g')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='h', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='i')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='j')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='k')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='l')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='m')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='n', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='o')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='p')

    # AVGPOOL 
    X = AveragePooling2D((2,2),name= 'avg_pool')(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (64, 64, 1), classes = 2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 50, batch_size = 32)

Loss, test_acc = model.evaluate(X_test, y_test)

y_pred = np.argmax(model.predict(X_test), axis=1)

y_true = np.argmax(y_test,axis = 1)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true,y_pred,beta = 2, average = 'binary')

print("Precision: %f - Recall - %f - fbeta_score: %f" %(precision,recall,fbeta_score))

