#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import random

random.seed(1337)
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
import argparse
import os

from common import load, read_mapping_file

DESCRIPTION = """
Runs a CNN on the offline preprocessed lung data
"""

BATCH_SIZE = 1
NB_CLASSES = 2
NB_EPOCH = 3

# input image dimensions
INPUT_SHAPE = (1, 120, 120, 120)
# number of convolutional filters to use
NB_FILTERS = 32
# size of pooling area for max pooling
POOL_SIZE = (2, 2, 2)
# convolution kernel size
KERNEL_SIZE = (5, 5, 5)


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('-m', '--mapping_file', help='<PATH> To the sample mapping file', type=str, required=True)
    parser.add_argument('-s', '--save', help='<Path> to save the model', type=str, required=True)
    return parser


def generate_training_set(training_set, input_folder):
    batch_samples = np.zeros((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]))
    batch_features = np.zeros((BATCH_SIZE))

    true_postive_set = training_set['cancer'] == 1
    x = 0
    while True:
        for i in range(BATCH_SIZE):
            if random.uniform(0, 1) >= .5:
                index = random.choice(training_set[true_postive_set].index)
            else:
                index = random.choice(training_set[~true_postive_set].index)
            mat_pth = os.path.join(input_folder, "%s.npz" % training_set.ix[index]['id'])
            img = load(mat_pth).astype(np.float32)
            img = img.reshape(1, *INPUT_SHAPE)
            batch_samples[i] = img
            batch_features[i] = training_set.ix[index]['cancer']
        batch_features = np_utils.to_categorical(batch_features.astype(np.int), NB_CLASSES)
        x += 1
        print(x)
        yield (batch_samples.astype(np.float32), batch_features)


def generate_test_set(test_set, input_folder):
    x = 0
    print(test_set['cancer'])
    while True:
        for i, row in test_set.iterrows():
            mat_pth = os.path.join(input_folder, "%s.npz" % row['id'])
            sample = load(mat_pth).astype(np.float32)
            sample = sample.reshape(1, *INPUT_SHAPE)
            feature = np.zeros((1))
            feature[0] = row['cancer']
            feature = np_utils.to_categorical(feature.astype(np.int), NB_CLASSES)
            x += 1
            print(x)
            yield (sample, feature)


def build_network():
    model = Sequential()

    model.add(Convolution3D(NB_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2],
                            border_mode='valid',
                            input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=POOL_SIZE))
    model.add(Convolution3D(NB_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=POOL_SIZE))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


# Driver function
def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    df_mapping = read_mapping_file(args.mapping_file)

    infiles = os.listdir(args.input)
    infiles = [infile[:-4] for infile in infiles]

    all_files = df_mapping[df_mapping['id'].isin(infiles)]

    training_mask = np.zeros((all_files.shape[0]))

    training_mask[np.random.uniform(0, 1, all_files.shape[0]) <= .9] = 1
    training_mask = training_mask.astype(bool)

    if os.path.exists(args.save):
        model = load_model(args.save)
    else:
        model = build_network()
        model.fit_generator(generate_training_set(all_files[training_mask], args.input), int(len(infiles)/BATCH_SIZE), NB_EPOCH)
        save_model(model, args.save, overwrite=True)

    score = model.evaluate_generator(generate_test_set(all_files[~training_mask], args.input), 250)

    print(model.predict_generator(generate_test_set(all_files[~training_mask], args.input), 250))
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

# Used for thread safety
if __name__ == '__main__':
    main()
