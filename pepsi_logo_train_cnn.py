# import the necessary packages
import imutils,cv2, time, os, sys
import numpy as np
import argparse
from imutils import paths
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

def pepsi_logo_cnn(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)

    model.add(Conv2D(8, (7, 7), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Conv2D(16, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#    model.add(BatchNormalization())

    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (7, 7), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])
    # return the constructed network architecture
    return model

##########################################################################################
##########################################################################################
##########################################################################################
def main(path):
    #path = '/home/rizwan/xplain/train_images2/'
    (imgW, imgH) = (128, 128)
    data = []
    labels = []
    EPOCHS = 50
    im_chanel=3
    classes=2
    mini_batch = 100
    #kernel = np.ones((2, 2), np.uint8)

    # loop over the image paths in the training set
    print("[INFO] Loading images...")

    for imagePath in paths.list_images(path):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
        data.append(image)
        if os.path.split(imagePath)[1].startswith('p'):
            labels.append(1)
        else:
            labels.append(0)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.20, random_state=19)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)
    trainX,  trainY = shuffle(trainX, trainY, random_state=19)
    testX,  testY = shuffle(testX, testY, random_state=19)
    aug = ImageDataGenerator(rotation_range=1, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,
        horizontal_flip=False, fill_mode="nearest")
    #labels = np.array(labels).ravel(-1)


    print("[INFO] training classifier...")

    early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
    mdl_save = ModelCheckpoint('pepsi_model', save_best_only=True, monitor='val_acc', mode='auto')

    model=pepsi_logo_cnn(imgW, imgH, im_chanel, classes)
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=mini_batch),callbacks=[early_stop, mdl_save],
        validation_data=(testX, testY), steps_per_epoch=len(trainX),epochs=EPOCHS, verbose=2)

    #########################################################################################
    #########################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainpath", required=True, type=str)
    args = parser.parse_args()
    trainpath = os.path.abspath(args.trainpath)

    main(trainpath)
