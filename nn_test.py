'''Trains a simple deep NN on the Dogs vs Cats dataset.
Gets to 60.2% test accuracy after 30 epochs
(there is *a lot* of margin for parameter tuning).
1.33 seconds per epoch on a RTX 2018 Ti GPU.
'''

import numpy as np
import cv2
import os
import gc
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.advanced_activations import ReLU
from keras.activations import tanh, sigmoid
from keras.models import Model
from keras.optimizers import SGD
import sys
from skimage import io
import warnings

np.random.seed(1337)  # for reproducibility

batch_size = 64
nb_classes = 2
nb_epoch = 30


def paths_list_from_directory(directory):

    subdirs = next(os.walk(directory))[1]  # returns list of immediate children of parent directory

    path_list = []
    # for folders in the list of subdirectories, add all the files within it to the path list
    for folder in subdirs:
        # loop over files and make a list of full paths then add to the path list
        file_names = [os.path.join(directory, folder, f) for f in os.listdir(os.path.join(directory, folder))]
        path_list += file_names

    return path_list


def load_image(filename):

    # [1] Get the file category and make the conversion. If 'dog' assign it integer 1, if 'cat' assign it integer 0.
    # check if string 'Dog' is in the path name (in the folder named Dog)
    if 'Dog' in filename:
        label = 1
    else:
        label = 0

    # [2] Load the image in greyscale with opencv.
    # use cv2 image read function, and greyscale parameter
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_dim var
    # get values of height and width from image shape
    height, width = image.shape[:2]

    # create a tuple that flags 1 on the larger dimesion, height in position 0 and width in position 1
    if height < width:
        crop_dim = (0, 1)
    else:
        crop_dim = (1, 0)

    # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.
    # use our crop_dim tuple in calculation to determine what the margin (distance from edge of image)
    margin = int(crop_dim[0] * (height-width)/2 + crop_dim[1] * (width-height)/2)
    # use the crop_dim and calculated margin to exclude the margin from the appropriate dimension
    image = image[(crop_dim[1]*margin):(width - crop_dim[1]*margin), (crop_dim[0]*margin):(height-crop_dim[0]*margin)]

    # [5] Resize the image to 48 x 48 and divide it with 255.0 to normalise it to floating point format.
    # use opencv resize function
    image = cv2.resize(image,(48, 48))

    # set the image as a float type and divide by 255.0 to normalize
    image.astype(float)
    image = image/255.0

    return image, label


# noinspection PyPep8Naming,PyUnusedLocal
def DataGenerator(img_addrs, batch_size, num_classes):

    # We removed the img_labels from DataGenerator as the load_image returns a label from the path anyway
    while 1:
        # Ensure randomisation per epoch using np.random.shuffle
        # addrs_labels = list(zip(img_addrs, img_labels))
        np.random.shuffle(img_addrs)
        # img_addrs, img_labels = zip(*addrs_labels)

        X = []
        Y = []

        # we removed count var from here, loop over range of images
        for j in range(len(img_addrs)):

            # [1] Call the load_images function and append the image in X.
            image, label = load_image(img_addrs[j])
            #### used to have resize
            X.append(image)
            # [2] Create a one-hot encoding with np.eye and append the one-hot vector to Y.
            Y.append(np.eye(num_classes)[label])  # takes the label from the load_image function and makes 1-hot.

        # [3] Commpare the count and batch_size (hint: modulo operation) and if so:
        # we use j+1 as count var was previously equivalent to this
        # when we hit a value of j+1 that is a multiple of batch size we run below block
            if ((j+1) % batch_size) == 0:
                #   - Use yield to return X,Y as numpy arrays with types 'float32' and 'uint8' respectively
                X = np.array(X, dtype=np.float32)
                X = X.reshape(batch_size, 2304)
                Y = np.array(Y, dtype=np.uint8)

                yield X, Y

                #   - delete X,Y
                del X
                del Y

                #   - set X,Y to []
                X = []
                Y = []

                # garbage collect
                gc.collect()


if __name__ == "__main__":

    # pull paths from function and shuffle these before splitting to train and val
    paths = paths_list_from_directory('PetImages')
    np.random.shuffle(paths)

    # Use train test split to split to default 0.75 train and 0.25 test
    train, val = train_test_split(paths)

    # Below section was for data creation without the generator so is no longer needed.
    # X_train = []
    # Y_train = []

    # for p in train:
    #    tpl = load_image(p)
    #    X_train.append(tpl[0])
    #    Y_train.append(np.eye(nb_classes)[tpl[1]])
    #
    # X_val = []
    # Y_val = []
    #
    # for p in val:
    #     tpl = load_image(p)
    #     X_val.append(tpl[0])
    #     Y_val.append(np.eye(nb_classes)[tpl[1]])

    # lt = len(X_train)
    # lv = len(X_val)

    # X_train = np.asarray(X_train)
    # Y_train = np.asarray(Y_train)
    # X_val = np.asarray(X_val)
    # Y_val = np.asarray(Y_val)

    # X_train = X_train.reshape(lt, 2304)
    # X_val = X_val.reshape(lv, 2304)

    # Build the network structure. We tried using tanh and sigmoid activation (non-linear) on hidden layers...
    # ...but this did not improve results
    inputs = Input(shape=(2304,))
    x = inputs

    i = 0

    while i < 5:
        x = Dense(254)(x)
        x = ReLU()(x)  # Non-linearily
        #x = Activation(sigmoid)(x)
        x = Dropout(0.5)(x)
        i = i + 1

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=inputs, output=predictions)

    # gradient decent parameters
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print a summary of the model
    model.summary()

    # fit the model with our created generator, validate on the generator with the validation data over 10 batches
    history = model.fit_generator(DataGenerator(train, batch_size, nb_classes), epochs=nb_epoch,
                                  steps_per_epoch=64, verbose=1,
                                  validation_data=DataGenerator(val, batch_size, nb_classes),
                                  validation_steps=10)

    # score by doing a full run over the validation set
    score = model.evaluate_generator(DataGenerator(val, len(val), nb_classes), verbose=0, steps=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Added a quick and dirty interface to take images from the validation set, display them and their predicted label
    idx = 0
    while 1:
        cv2.destroyAllWindows()
        # read original image and use load_image to get the greyscale cropped and resized input image to the network
        image = cv2.imread(val[idx])
        image_gs = load_image(val[idx])

        # process the image into expected network input
        x_pred = np.array(image_gs[0])
        x_pred = x_pred.reshape(1, 2304)

        # predict from the model
        label = model.predict(x_pred)

        # set label to the most likely category (Good boi! = Dog, of course)
        if label[0][0] > 0.5:
            label = 'Cat'
        else:
            label = 'Good boi!'

        # Print the classification on the image
        cv2.putText(image, label, (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)

        # Display the image with label printed on it
        cv2.imshow("Image", image)
        keypress = cv2.waitKey(0)
        # Set to scroll through the dataset using 'd' to move forward, 'a' to move back and esc to exit
        if keypress == 97 and idx > 0:
            idx -= 1
        elif keypress == 100 and idx < len(paths):
            idx += 1
        elif keypress == 27:
            break
        else:
            pass
