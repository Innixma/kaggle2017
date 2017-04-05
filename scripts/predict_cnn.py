#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import random

random.seed(1337)
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
import argparse
import os

from common import load, read_mapping_file

DESCRIPTION = """
Runs a CNN on the offline preprocessed lung data
"""

BATCH_SIZE = 10
NB_CLASSES = 2
NB_EPOCH = 3

# input image dimensions
INPUT_SHAPE = (1, 120, 120, 120)
# number of convolutional filters to use
NB_FILTERS = 32
# size of pooling area for max pooling
POOL_SIZE = (2, 2, 2)
# convolution kernel size
KERNEL_SIZE = (3, 3, 3)


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('-m', '--mapping_file', help='<PATH> To the sample mapping file', type=str, required=True)
    return parser


def generate_arrays_from_file(training_set, input_folder):
    while 1:
        rows = training_set.ix[random.sample(training_set.index, BATCH_SIZE)]
        X = []
        for i, row in rows.iterrows():
            mat_pth = os.path.join(input_folder, "%s.npz" % row['id'])
            img = load(mat_pth).astype(np.float32)
            img = img.reshape(*INPUT_SHAPE)
            X.append(img)
        X = np.array(X).reshape(BATCH_SIZE, *INPUT_SHAPE)
        print(X.shape)
        yield (X, rows['cancer'].values.astype(np.int8).reshape(BATCH_SIZE))


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
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
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

    training_set = df_mapping[df_mapping['id'].isin(infiles)]
    # for (train, data) in generate_arrays_from_file(training_set, args.input):
    #     print(train)
    model = build_network()
    # print(model.summary())
    #
    model.fit_generator(generate_arrays_from_file(training_set, args.input), BATCH_SIZE, NB_EPOCH)
    #
    # df_mapping[]

# Used for thread safety
if __name__ == '__main__':
    main()
