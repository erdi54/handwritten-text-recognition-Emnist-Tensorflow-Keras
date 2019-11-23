import numpy as np
import os
import cv2
from mnist.loader import MNIST
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.backend import get_session
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM
from keras import backend as K
from keras.constraints import maxnorm
from keras.utils import np_utils

def resh(ipar):
    opar = []
    for image in ipar:
        opar.append(image.reshape(-1))
    return np.asarray(opar)



class ModelTrain:
    def __init__(self):
        self.model = None
        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None
        return

    def load_data(self):
        mndata = MNIST('data')
        # This will load the train and test data

        x_train, y_train = mndata.load('./data/character/emnist-byclass-train-images-idx3-ubyte',
                                                     './data/character/emnist-byclass-train-labels-idx1-ubyte')
        x_test, y_test = mndata.load('./data/character/emnist-byclass-test-images-idx3-ubyte',
                                                   './data/character/emnist-byclass-test-labels-idx1-ubyte')
        # Convert data to numpy arrays and normalize images to the interval [0, 1]
        x_train = np.array(x_train)/ 255.0
        y_train = np.array(y_train)

        x_test = np.array(x_test)/ 255.0
        y_test = np.array(y_test)

        # Reshaping all images into 28*28 for pre-processing
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 28, 28)

        # for train data
        for t in range(697932):
            x_train[t] = np.transpose(x_train[t])

        # for test data
        for t in range(116323):
            x_test[t] = np.transpose(x_test[t])

        print('Process Complete: Rotated and reversed test and train images!')

        # Reshaping train and test data again for input into model
        x_train = x_train.reshape(x_train.shape[0], 784, 1)
        x_test = x_test.reshape(x_test.shape[0], 784, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = resh(x_train)
        x_test = resh(x_test)

        # one-hot encoding:
        y_train = np_utils.to_categorical(y_train,62)
        y_test = np_utils.to_categorical(y_test,62)


        print('EMNIST data loaded: train:', len(x_train), 'test:', len(x_test))
        print('Flattened X_train:', x_train.shape)
        print('Y_train:', y_train.shape)
        print('Flattened X_test:', x_test.shape)
        print('Y_test:', y_test.shape)

        self.train_images = x_train
        self.test_images = x_test
        self.train_labels = y_train
        self.test_labels = y_test

    def creatModel(self):
        K.set_learning_phase(1)
        self.model = Sequential()
        self.model.add(Reshape((28,28,1), input_shape=(784,)))
        # add the layer below for an accuracy of 89%.(Training time - over 20 hours)
        self.model.add(Convolution2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', padding='same',
                                kernel_constraint=maxnorm(3)))
        self.model.add(Convolution2D(32, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(62, activation='softmax'))

        opt = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(self.model.summary())

        return self.model

    def train(self, model):
        # Set a learning rate reduction
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        get_session().run(tf.global_variables_initializer())

        history = self.model.fit(self.train_images, self.train_labels, validation_data=(self.test_images, self.test_labels), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)
        # evaluating model on test data. will take time
        scores = model.evaluate(self.test_images, self.test_labels, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # saves the model info as json file
        model.save_weights("model.h5")

        # Creates a HDF5 file 'model.h5'
        return history

    def letters_extract(self, img_path=None, out_size=28):

        im_num = 0
        list_path = os.listdir(img_path)
        l = [img_path + object for object in list_path if ".jpg" or ".png" in object]

        # enter input image here
        img = cv2.imread(l[im_num])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

        # Get contours
        _,contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output = img.copy()

        letters = []
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                letter_crop = gray[y:y + h, x:x + w]
                # print(letter_crop.shape)

                # Resize letter canvas to square
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    # Enlarge image top-bottom
                    # ------
                    # ======
                    # ------
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    # Enlarge image left-right
                    # --||--
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                # Resize letter to 28x28 and add letter and its X-coordinate
                letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

        # Sort array in place by X-coordinate
        letters.sort(key=lambda x: x[0], reverse=False)
        return letters

    def predict_image(self,model,img):
        # The original EMNIST dataset has 62 different characters (A.Z, 0..9, etc.).
        emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                         79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                         107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

        img_arr = np.expand_dims(img, axis=0)
        img_arr = 1 - img_arr / 255.0
        img_arr[0] = np.rot90(img_arr[0], 3)
        img_arr[0] = np.fliplr(img_arr[0])
        img_arr = img_arr.reshape((1, 784))

        result = model.predict_classes([img_arr])
        return chr(emnist_labels[result[0]])

    def img_to_str(self,model,img_path,letters):
        letters = self.letters_extract(img_path)
        s_out = ""
        for i in range(len(letters)):
            dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
            s_out += self.predict_image(model, letters[i][2])
            if (dn > letters[i][1] / 4):
                s_out += ' '
        return s_out














































