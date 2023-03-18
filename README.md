# Tensorflow-TransferLearning



## Overview



Transfer Learning is a machine learning method where we reuse a model trained on a first dataset called the source dataset as the starting point for training a model on a second dataset called the target dataset.

  In this project, the source dataset is a large dataset like ImageNet and the target dataset is a much smaller dataset is **5 flower categories**.


## Transfer Learning

 - Take a slice of layers from a previously trained model.
 - Freeze their weights, so as to avoid destroying any of the
   information they contain during future training rounds on your new
   dataset.
 - Add some new, trainable layers on top of the frozen layers. They will
   learn to turn the old features into predictions on a new dataset.
 - Train the new layers on your new dataset.
 - Unfreezing the entire model obtained above and re-training it on the
   new data with a very low learning rate

The last step is known as **Fine Tuning**.


## Approach

|||
|-|-|
|Network Architecture|[MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/)|
|Target Dataset| [5 classes of flower dataset](https://github.com/hikariakio/Tensorflow-TransferLearning/tree/main/dataset) |
|Compiler|[SGD optimizer](https://keras.io/api/optimizers/sgd/)|
|Fine Tuning| NOT INCLUDED

 <!-- LICENSE -->
## License

Distributed under the MIT License.

## Acknowledgement

* Dataset Images are from tf_flower