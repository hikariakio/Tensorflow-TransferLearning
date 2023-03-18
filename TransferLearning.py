# setup
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
import PIL.Image
import pathlib
from math import ceil, floor

def Task1(folder_name):
    '''
          Download dataset from a folder directory.

          Params:
              - folder_name : relative folder directory

          Returns:
              - full_path directory , total number of images

    '''
    data_dir = pathlib.Path(folder_name).absolute()
    image_count = len(list(data_dir.glob('*/*.jpg')))
    return data_dir, image_count

def Task2():
    '''
            Download pretrained mobilenetv2

            Returns:
              - mobilenetv2 model
    '''
    base_model = tf.keras.applications.MobileNetV2()
  
    return base_model

def Task3(base_model):
    '''
        Task3 - removing the last layer in netV2 and replace with custom dense layer

        Params:
          - base model of mobilenetv2

        Returns:
          - a new model with frozen weights having a fresh output layer
    '''
    #remove the output layer from the network
    frozen_layers = tf.keras.Model(base_model.input,base_model.layers[-2].output)
    #freeze weights of remaining layers
    frozen_layers.trainable = False

    #change pixel from [0,255] into [-1,1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # average spatial dimensions to one vector
    # global_average_layer = tf.keras.layers.GlobalAveragePooling2D() # already included in the frozen layer

    # output layer with 5 classes
    prediction_layer = tf.keras.layers.Dense(5) 

    # regularization to reduce overfitting 
    # dropout_layer = tf.keras.layers.Dropout(0.2) # decided not to use

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = frozen_layers(x, training=False)
    outputs = prediction_layer(x) # TODO?: "activation:softmax"
    model = tf.keras.Model(inputs, outputs)

    return model

def split_dataset(dataset,split_amount):
    '''
        AUX function - split dataset in split_amount batch size

         Params:
          - dataset - dataset to be splitted
          - split_amount - batch size to split

        Returns:
          - left_set will take split_amount of the dataset
          - right_set will take remaning amount. 
            (if left_set takes everything from dataset, right_set will be empty)
    '''
    batchsize = len(dataset)

    left_set = dataset.take( split_amount )
    right_set = dataset.skip( split_amount )
    return left_set ,right_set

def Task4(data_dir,train_perc=0.75,val_perc = 0.15, test_perc = 0.1 , rand_seed = 123):
    '''
        Preparing Dataset to training,validation and testing set.

        Params:
          data_dir - full datadirectory
          train_perc,val_perc,test_perc - amount of dataset , sum of them must be 1
          rand_seed - random seed

        Returns:
          train set, validation set , test set
          class_names - list containing class names
    '''
    if(train_perc + val_perc + test_perc != 1):
        raise Exception ("Dataset spltting went wrong, check DIRECTORY or Percent")
    dataset = tf.keras.utils.image_dataset_from_directory(
                    data_dir,                    
                    shuffle = True,
                    seed=rand_seed,
                    image_size=IMG_SIZE,
                    batch_size=BATCH_SIZE)

    class_names = dataset.class_names

    total_size = len(dataset)

    train_size = floor(total_size  * train_perc)
    remain_size = total_size - train_size

    val_size = floor(remain_size * val_perc /(val_perc + test_perc))

    train_dataset,val_dataset = split_dataset(dataset, train_size )
    val_dataset, test_dataset = split_dataset(val_dataset,val_size)

    print(class_names)
    print("Training Batch Size   : " , len(train_dataset))
    print("Validation Batch Size : " , len(val_dataset))
    print("Testing Batch Size    : " , len(test_dataset))

    return train_dataset,val_dataset,test_dataset,class_names

def Task5_compile(model,learning_rate , momentum):
  '''
    Compiling a model using SGD optimizer

    Params:
      learning_rate , momentum

    Returns:
      model - a compiled model using above parameters
  '''
  
  optimizer =tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=False)

  model.compile(optimizer = optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  return model

def Task5_train(model, train_ds,val_ds,epoch = 10):
  '''
      Fitting a model with using given train dataset and validation dataset in given epoch

      Params:
        train_ds - training dataset
        val_ds - validation dataset
        epoch - iterations times need to perform

      Returns
        history - a full log contained acc and loss.
  '''
  history = model.fit(train_ds,
                    epochs=epoch,
                    validation_data=val_ds)
  return history

def Task5_train(model, train_ds,val_ds,epoch = 10):
  '''
      Fitting a model with using given train dataset and validation dataset in given epoch

      Params:
        train_ds - training dataset
        val_ds - validation dataset
        epoch - iterations times need to perform

      Returns
        history - a full log contained acc and loss.
  '''
  history = model.fit(train_ds,
                    epochs=epoch,
                    validation_data=val_ds)
  return history

def Task6(history):
  '''
      Plotting a given history

      Params:
        history - a full log contained acc and loss
  '''
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  learning_rate = str(history.model.optimizer.learning_rate.numpy())
  momentum = str (history.model.optimizer.momentum.numpy())
  fig = plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)

  plt.suptitle(f'Learning Rate = {learning_rate} , Momentum = {momentum}',fontsize = 14 , y = 1.05)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy',fontsize = 8)
  plt.ylim([0.2,1.0])
  plt.xlabel('Epoch',fontsize = 8)
  plt.title('Training and Validation Accuracy',fontsize = 8)

  plt.subplot(1, 2, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('SparseCategorical CrossEntropy',fontsize = 8)
  plt.ylim([0,2.0])
  plt.title('Training and Validation Loss',fontsize = 8)
  plt.xlabel('Epoch',fontsize = 8)
  plt.show()

def TestIndividualImage(model,filename):
  test_img = PIL.Image.open(filename).convert('RGB')
  test_img = test_img.resize((224,224))

  pix = np.array(test_img)
  plt.imshow(pix)

  f_data =  np.expand_dims(pix,0)
  y_pred = model.predict(f_data)
  print("PREDICTED" , class_names[np.argmax(y_pred)])


def Model_F():
    #include_top false -> we need to define our own imgshape.
    
    base_model = tf.keras.applications.MobileNetV2()
    # base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
    #                                                include_top=False,
    #                                                weights='imagenet')
    
    #remove the output layer from the network
    frozen_layers = tf.keras.Model(base_model.input,base_model.layers[-2].output)
   

    #change pixel from [0,255] into [-1,1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

   

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    outputs = frozen_layers(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def Model_N():

    # dropout_layer = tf.keras.layers.Dropout(0.2)
    prediction_layer = tf.keras.layers.Dense(5) 

    shape = (1280)

    inputs = tf.keras.Input(shape=shape)
    # x = dropout_layer(inputs)
    outputs = prediction_layer(inputs)

    model = tf.keras.Model(inputs, outputs)
    return model


def Task9(model_f,train_ds,val_ds,test_ds):
    '''
        Preparation of dataset for accelerated model

        params:
            model_f - F function
            train_ds,val_ds,test_ds - original dataset (x)

        returns
            three sets of F(x)
    '''
    train_batches = list(train_ds.unbatch().as_numpy_iterator())
    new_train_y = [y for x,y in train_batches]
    new_train_x = [x for x,y in train_batches]
    new_train_x = model_f.predict(np.array(new_train_x))



    val_batches = list(val_ds.unbatch().as_numpy_iterator())
    new_val_y = [y for x,y in val_batches]
    new_val_x= [x for x,y in val_batches]
    new_val_x = model_f.predict(np.array(new_val_x))


    test_batches = list(test_ds.unbatch().as_numpy_iterator())
    new_test_y = [y for x,y in test_batches]
    new_test_x = [x for x,y in test_batches]
    new_test_x = model_f.predict(np.array(new_test_x))

    new_train_ds = tf.data.Dataset.from_tensor_slices((new_train_x,new_train_y)).batch(32)
    new_val_ds = tf.data.Dataset.from_tensor_slices((new_val_x,new_val_y)).batch(32)
    new_test_ds = tf.data.Dataset.from_tensor_slices((new_test_x,new_test_y)).batch(32)

    return new_train_ds,new_val_ds,new_test_ds

# *************** GLOBAL VARIABLES ( TWEAK FOR SETTING) *********************** #

BATCH_SIZE = 32
#default for v2 is 224,224
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,) # RGB
DATASET_DIRECTORY = 'small_flower_dataset'

# **************************************************************************************** #

if __name__ == "__main__":

    # SEE "GLOBAL VARIABLES" IF YOU WANT TO TWEAK SETTINGS
    
    full_dir , count = Task1(DATASET_DIRECTORY)

    mobile_net_v2_model = Task2()



    train_ds,val_ds,test_ds,class_names = Task4(data_dir = full_dir,
                                                train_perc = 0.75, 
                                                val_perc = 0.15 , 
                                                test_perc = 0.1)


    # **** normal transfer learning - change following section for different learning_rate and momentumm *** #
    # Task 3,5,6,7,8

    learning_rate = 0.01
    momentum = 0.0
    epoch = 15

    custom_model = Task3(mobile_net_v2_model)
    transfer_learning_model = Task5_compile(model = custom_model,learning_rate = learning_rate,momentum = momentum)
    transfer_learning_history = Task5_train(model = transfer_learning_model,train_ds = train_ds,val_ds = val_ds , epoch = epoch)
    Task6(transfer_learning_history)
    transfer_learning_model.evaluate(test_ds)

    # EXTRA (if you want to predict an image using the model) #
    # TestIndividualImage( model = transfer_learning_model, filename = 'a.jpg')
    
    model_f = Model_F()
    new_train_ds,new_val_ds,new_test_ds = Task9(model_f=model_f,train_ds=train_ds,val_ds=val_ds,test_ds=test_ds)
    model_n = Model_N()

    # **** accelerated learning - change following section for different learning_rate and momentumm *** #
    # Task 10

    learning_rate = 0.1
    momentum = 0.6
    epoch = 15

    accelerated_learning_model = Task5_compile(model = model_n,learning_rate = learning_rate,momentum = momentum)
    accelerated_learning_history = Task5_train(model = accelerated_learning_model,train_ds = new_train_ds,val_ds = new_val_ds , epoch = epoch)
    Task6(accelerated_learning_history)
    accelerated_learning_model.evaluate(new_test_ds)