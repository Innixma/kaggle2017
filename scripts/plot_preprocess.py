#!/usr/bin/env python

# This patient is broken? Why?
# 0acbebb8d463b4b9ca88cf38431aac69.npz

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def plot_3d(image, threshold=-100):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


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
        patient_path = os.path.join(input_folder, patient)
        img = load(patient_path)
        print(img.min())
        print(img.max())
        print(patient)
        plot_3d(img, -.2)
    #
    # img = load(args.input)
    #
    #

# Used for thread safety
if __name__ == '__main__':
    main()
