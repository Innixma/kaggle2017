# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import ResNet50


from common import shuffle_weights


if __name__ == '__main__':
    model = ResNet50(include_top=True, weights='imagenet')

    img_path = "D:\\Projects\\lung\\kaggle2017\\scripts\\f0a6df_fb9d52a7217a45feb10f1640986ae704.png"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    print(model.summary())

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
