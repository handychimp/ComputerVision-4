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
from keras.activations import tanh
from keras.models import Model
from keras.optimizers import SGD

np.random.seed(1337)  # for reproducibility

batch_size = 64
nb_classes = 2
nb_epoch = 10


def paths_list_from_directory(directory):

    subdirs = next(os.walk(directory))[1]  # takes the immediate children of parent directory
    print("Sub directories: ", end="")
    print(*subdirs, sep=", ")
    path_list = []
    for folder in subdirs:
        # loop over files and make a list of tuples with label and path then add to the path list
        file_names = [os.path.join(directory, folder, f) for f in os.listdir(os.path.join(directory, folder))]
        path_list += file_names

    return path_list


def load_image(filename):

    # [1] Get the file category and make the conversion. If 'dog' assign it integer 1, if 'cat' assign it integer 0.
    if 'Dog' in filename:
        label = 1
    else:
        label = 0

    # [2] Load the image in greyscale with opencv.
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape[:2]
    crop_dim = None

    # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_drim var
    if height < width:
        crop_dim = (0, 1)
    else:
        crop_dim = (1, 0)

    # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.
    margin = int(crop_dim[0] * (height-width)/2 + crop_dim[1] * (width-height)/2)
    # print("margin size: " + str(margin))
    image = image[(crop_dim[1]*margin):(width - crop_dim[1]*margin),(crop_dim[0]*margin):(height-crop_dim[0]*margin)]

    # [5] Resize the image to 48 x 48 and divide it with 255.0 to normalise it to floating point format.
    image = cv2.resize(image,(48,48))

    return image, label


# noinspection PyPep8Naming,PyUnusedLocal
def DataGenerator(img_addrs, img_labels, batch_size, num_classes):

    while 1:
        # Ensure randomisation per epoch
        addrs_labels = list(zip(img_addrs, img_labels))
        np.random.shuffle(addrs_labels)
        img_addrs, img_labels = zip(*addrs_labels)

        X = []
        Y = []

        # count = 0

        for j in range(len(img_addrs)):

            # [1] Call the load_images function and append the image in X.
            image = load_image(img_addrs[j])
            X.append(image[0])
            # [2] Create a one-hot encoding with np.eye and append the one-hot vector to Y.
            # Y.append(np.eye(num_classes)[image[1]])
            Y.append(img_labels[j])
            # count += 1

        # [3] Commpare the count and batch_size (hint: modulo operation) and if so:
            if ((j+1) % batch_size) == 0:
                print("I yield!")
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
    # paths = paths_list_from_directory('./PetImages')

    # for p in paths:
    #    try:
    #        load_image(p)
    #    except AttributeError:
    #        print('Invalid Image...Deleting...')
    #        os.remove(p)
    #    except cv2.error:
    #        print('Invalid Image in OpenCV...Deleting...')
    #        os.remove(p)

    paths = paths_list_from_directory('PetImages')
    np.random.shuffle(paths)

    # Use train test split
    train, val = train_test_split(paths)

    X_train = []
    Y_train = []

    for p in train:
        tpl = load_image(p)
        X_train.append(tpl[0])
        Y_train.append(np.eye(nb_classes)[tpl[1]])

    X_val = []
    Y_val = []

    for p in val:
        tpl = load_image(p)
        X_val.append(tpl[0])
        # CHECK THIS BIT
        Y_val.append(np.eye(nb_classes)[tpl[1]])

    lt = len(X_train)
    lv = len(X_val)

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_val = np.asarray(X_val)
    Y_val = np.asarray(Y_val)

    X_train = X_train.reshape(lt, 2304)
    X_val = X_val.reshape(lv, 2304)

    inputs = Input(shape=(2304,))
    x = inputs

    i = 0

    while i < 3:
        x = Dense(254)(x)
        # x = tanh()(x)  # Non-linearily
        x = Activation(tanh)(x)
        x = Dropout(0.5)(x)
        i = i + 1

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=inputs, output=predictions)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    #history = model.fit(X_train, Y_train,
    #                    batch_size=batch_size, epochs=nb_epoch,
    #                    verbose=1, validation_data=(X_val, Y_val))
    history = model.fit_generator(DataGenerator(train, Y_train, batch_size, nb_classes), epochs=nb_epoch,
                                  steps_per_epoch=50, verbose=1,
                                  validation_data=(X_val, Y_val))
    # score = model.evaluate(X_val, Y_val, verbose=0)
    score = model.evaluate_generator(DataGenerator(val, Y_val, len(X_val), nb_classes), verbose=0, steps=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Play times! Add something to play with the model.
    # np.random.shuffle(paths)
    idx = 0
    np.random.shuffle(val)
    while 1:
        cv2.destroyAllWindows()
        image = cv2.imread(val[idx])
        image_gs = load_image(val[idx])
        x_pred = np.array(image_gs[0])
        x_pred = x_pred.reshape(1, 2304)
        label = model.predict(x_pred)

        if label[0][0] == 1:
            label = 'Cat'
        else:
            label = 'Dog'

        # Print the classification on the image
        cv2.putText(image, label, (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Image", image)
        keypress = cv2.waitKey(0)
        # if key back, key forward, esc
        if keypress == 97 and idx > 0:
            idx -= 1
        elif keypress == 100 and idx < len(paths):
            idx += 1
        elif keypress == 27:
            break
        else:
            pass
