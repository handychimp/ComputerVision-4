# Introduction to Neural Computations - Lab session

### About

* The aim of this session is to provide you with a practical understanding of neural networks

The main tasks are listed as follows:

- [ ] Start with the `Keras` Deep Learning library and build upon given code.

- [ ] Experiment with different neural architectures on the [Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset and improve network fine-tuning.

- [ ] Learn how to save models in JSON format and how to save model weights in .h5 files.

- [ ] Create a `DataGenerator` function that takes a list (or array) of files and returns the loaded files in array types alongside their labels.

### A few Words about Keras

![](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

Keras is not a stand-alone Python library but rather an API running on top of other Deep Learning frameworks. In this course you will be using a `Tensorflow` backend. - In case you are unsure about the backend that you are using after you downloaded all the required libraries, you should locate the (hidden) keras.json file in your home directory:
```
$HOME/.keras/keras.json
```

You can then access it with your favourite editor and it should have default values such as:
```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
The most important ones are:
- the `image_data_format` which specifies the format that you will be using when processing data. You can choose between having the RGB channels first or last.
- `floatx` which is the default variable type that data will take within your network. **If along the way you have memory problems, you can decrease the floating point to e.g. 16-bits and half the memory requirements.**
- The backend variable `backend` which should be tensorflow.

---

### Dependencies
 A full list of library dependencies can be found below:
 ```
$ pip install numpy opencv-python h5py keras tensorflow matplotlib pandas
$ pip install -U scikit-learn
 ```
 Optional dependencies:
 ```
$ pip install Pillow scikit-image,
 ```

### The Kaggle Pets Dataset

 For this practical session, all your tasks will be based on the simple two-class dataset Dogs vs Cats. The dataset can be downloaded from [this link](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The training archive contains 25,000 images.

 ! There may be some corrupted files, so do take a couple of minutes and discard any unusable images !

 Because we are interested on the inner-workings of neural networks, it is suggested to resize the images to size 48x48 (more about image processing will be explained later in this session).


 ### Getting Started

 Open the Python file `nn_test.py` and have a look at the first few lines of code.

 #### Preparing and processing training data

 Start by completing the missing code in function `paths_list_from_directory(directory)`. In order to properly evaluate the performance of trained models, we will also need an evaluation set. Therefore, after you finished loading the filepaths, make use of `scikit-learn` [train_test_split function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and return your list of training and validation files

 Moving forward, we will be required to define a function to load the images. Because the dataset is (relatively) small, we will load them directly in a list as `numpy` arrays. So your function should look like this:
 ```
def load_image(filename):

  # [1] Get the file category and make the conversion. If 'dog' assign it integer 1, if 'cat' assign it integer 0.

  # [2] Load the image in greyscale with opencv.

  height, width = image.shape[:2]
  crop_dim = None

  # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_drim variable.

  # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.

  # [5] Resize the image to 48 x 48 and divide it by 255.0 to normalise it to floating point format.

  return (image,label)

 ```

#### Creating a Keras Model

When building a model, you can choose between two APIs:
 - [Sequential](https://keras.io/models/sequential/), which is a 'linear stack of layers' with the model being initialised with the `Sequential()` function. To then add layers, you can invoke the `.add()` method of the Model class. You can specify the input shape (i.e. the shape of your data similarly as the `numpy.shape` function) by using the `input_shape` parameter in your first layer.
 - [Functional](https://keras.io/getting-started/functional-api-guide/), which is designed for more complex models as it provides greater flexibilty than the `Sequential` API. In this case no initial API declaration is needed and layers are linked by also passing them to the following layers (there will be more examples on this).

 We will use the `Functional` API as it will also be more useful for the assignments.

#### Input Layer
`Input()` is used to instantiate a Keras tensor.

Keras tensors are the essential blocks when building neural models that are simply defined by the first (input) and last (output) tensor of the model.
```
inputs = Input(shape=(2304,))
```
The shape parameter should be a tuple of integers that includes the shape of the data (e.g. width, height, # colour channels) but **does not** include the batch size.

In this case, our input size is a vector of size 2304 (48 x 48).

#### Hidden Layer(s)
The `Dense` layers are normal neurones. They essentially implement the operation `y = activation(dot(x, kernel) + bias)` where `activation` is the non-linearity that can be passed as an argument. `kernel` is the weight matrix corresponding to the layer and `bias` is the bias vector of the layer. You also need to specify the number of neurones that the network will include. In this example, we use 254 neurones per layer.

```
x = Dense(254)(x)
x = ReLU()(x)#Non-linearily
```
*Since you will be using images, you should note that in the case that the shape of the layer input is greater than 2, it will be flattened to a vector*

**So it is important to know EXACTLY the shapes of your tensors at each step**

#### Activation Functions

Activations should follow directly after each of your layers. The alternative is to also define them in the `keras.Layer` that you are using as a parameter. All core layers also include an activation as an optional parameter.
```
x = Dense(254, activation='relu')(x)
# OR
x = Dense(254)(x)
x = ReLU()(x)
```
#### Output Layer

In contrast with the `Input` layer, the output is defined by a normal `Dense` layer. Do keep in mind that each neurone in the output layer should correspond to a specific class so `# classes = # output neurones`.

To build a model, you will need to include all layers required in the computation of `predictions` given `inputs` (the `Model` class is specific to the Functional API)
```
predictions = Dense(2, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
```

#### Learning Algorithm

The optimiser is the backbone of every learning algorithm. It is the mathematical function based on which updates will be performed towards achieving an optimal model (by progressively minimising the objective or loss function).

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/03/06100746/grad.png)

In Keras you can call an optimiser based on its name alongside its (optional) parameters.

```
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
```

#### Compile the Model

Given that you have successfully created your model, you will then need to compile it with the `compile` method. The method takes two arguments:
- A loss function which is basically your feedback method on the model's performance.
- An optimiser (i.e. learning algorithm).

```
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
After compiling, you should have a configured neural network to deploy on your dataset.

#### Training on data

Before training the model, make use of the `model.summary()` method to visualise the network and get some insights on its shape, parameters etc.

It is very important to be sure that the network architecture is correct before attempting to train it.

The `.fit()` method trains the network for a given number of epochs. You will also need to specify the batch size (i.e. the number of examples that the network is trained with -> forward pass, before an update -> backpropagation). If `validation_data` is given, the method will also do a validation at the end of every epoch.

The `.evaluate()` will give a loss value and metrics score on the validation data. In both methods, you can specify the verbosity.

```
history = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_val, Y_val))
score = model.evaluate(X_val, Y_val, verbose=0)
```

Note your train and validation accuracies.

### Improving Model Accuracy

Finding the optimal architecture for a task is a complex, challenging and time-consuming. There is a plethora of literature on the subject with (what I find) the most interesting being [Neural Architectural Search (NAS)](https://arxiv.org/pdf/1611.01578.pdf).

Since there is a lot of margin for improving the network, try the following and (re-)record the accuracy and losses for each step:

- [ ] Modify the code to add one more Dense layer of 64 `relu` units and record the score.
- [ ] Modify the code so that you are able to add as many `Dense` layers of `relu` units as you want, controlled by a variable called `n_hidden_layers`.
- [ ] Add a `Dropout` layer with strength of 0.5 (This basically only considers half the inputs that it's given). We will also go back to Dropout and why it is useful later in the course.
- [ ] (Optional) play around with different scores and optimise on the number of layers, trying to find the optimal hyperparameters.


### Saving/Loading model/Weights

It is **VERY** important that you do not discard your model once you finish training. In order to present your results, you will need to have a copy of the learned weights that you obtained through training.

Keras allows you to store your model+weights or just the weights in `.HDF5` files. Hierarchical Data Format files (HDF) are file formats specifically designed to hold large data and are supported by most popular programming languages (including Java, Python, MATLAB & R). *- A bit of history: NASA was one of the first organisations to start using HDF files in mid-90s for Earth Observing Systems (EOS)*

To save your model after training, you can use the following command:
```
model.save(filepath)
```
And then you can respectively load it back with `model = load_model(filepath)`.

It is suggested however, for better memory management, to save the model in a JSON file and the weights at an HDF5 file.

```
json_string = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_string)
model.save_weights('my_model_weights.h5')
```

Also note that, when loading weights from previous models, you can also make use of the `by_name` flag in the `load_model()` method. This ensures that only layers with matching names are loaded in the new model.

**Keras Callbacks**

Keras also allows for callback functions to be used at each step of the training process. You can use callbacks to get a view on internal states and statistics of the model during training, or simply store the model in a directory.

It is extremely useful and highly recommended to used the `ModelCheckpoint()` function when training models. This will save your model (or just the weights) at the end of each epoch in an HDF5 file.

An example:
```
filepath_weights = 'weights-{epoch:02d}.hdf5'

checkpoint = ModelCheckpoint(filepath_weights, monitor='auto', verbose=0, save_best_only=False, mode='max', save_weights_only=True)

callbacks_list = [checkpoint]
model.fit(...,callbacks=callbacks_list)

```
Other useful callbacks:
- CSVLogger
- EarlyStopping
- ReduceLROnPlateau

(Or you can create your own)

### Using data generators

Python by itself provides a built-in generator method `yield` when running code in functions. This replaces `return` and allows to preserve the sequence of execution even after the function's lifetime, providing the result(s) to the caller without destroying local variables.

As `yield` is used in Python generators it would be nice to further study the notion of generator functions:

The main idea behind a data generator is to calculate a series or results iteratively (i.e. a list in which an operation is performed element-wise). However, in contrast to lists, generators have no length.

We use generators because modern computer vision datasets are very large and cannot always fit into RAM memory. Insead, by using generators, values are calculated on-the-fly without requiring to store any of the previous values.

So as a last task in this lab, you are requested to create a `DataGenerator` function  and make use of the `load_images`. You should submit this as Assignment 4.

```
def DataGenerator(img_addrs, img_labels, batch_size, num_classes):
  while 1:
    # Ensure randomisation per epoch
    addrs_labels = list(zip(img_addrs,img_labels))
    shuffle(addrs_labels)
    img_addrs, img_labels = zip(*addrs_labels)

    X = []
    Y = []

    count = 0

    for i in range(len(img_addrs)):

      # [1] Call the load_images function and append the image in X.

      # [2] Create a one-hot encoding with np.eye and append the one-hot vector to Y.

      count += 1

      # [3] Commpare the count and batch_size (hint: modulo operation) and if so:
      #   - Use yield to return X,Y as numpy arrays with types 'float32' and 'uint8' respectively
      #   - delete X,Y
      #   - set X,Y to []
      #   - use python garbage collector
```

You can then use the `.fit_generator()` method to use your `DataGenerator`. It is important to note that you will now be required to enter the number of steps manually as an additional method parameter.

(Optional) optimise your code further without including a `count` and only rely on `i`.
