#!/usr/bin/env python

# This patient is broken? Why?
# 0acbebb8d463b4b9ca88cf38431aac69.npz

import argparse
import os

import numpy as np

DESCRIPTION = """
Explain the script here
"""


def load(pth):
    return np.load(pth)['data']


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('-o', '--output', help='<PATH> The output folder', type=str, required=True)
    return parser


# Driver function
def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    # Some constants
    input_folder = args.input
    input_folder = os.path.abspath(input_folder)
    patients = os.listdir(input_folder)
    patients.sort()

    for patient in patients:
        if patient == '1c42a5da837f4122d2c3b59ca9b5f0fb.npz':
            patient_path = os.path.join(input_folder, patient)
            img = load(patient_path)
            print(img.min())
            print(img.max())
            print(patient)
            plot_3d(img, -.2)

# Used for thread safety
if __name__ == '__main__':
    main()
