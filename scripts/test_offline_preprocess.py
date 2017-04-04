#!/usr/bin/env python
from __future__ import print_function, division
import argparse

import numpy as np # linear algebra
import dicom
import os
import scipy.ndimage

from skimage import measure, morphology
import time

from common import save, load

DESCRIPTION = """
Explain the script here

Following the tutorial given here: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
Date accessed: 3/7/17
"""


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
    parser.add_argument('--min_bound', type=int, default=-1000)
    parser.add_argument('--max_bound', type=int, default=400)
    parser.add_argument('--pixel_mean', type=float, default=.25)
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('-o', '--output', help='<PATH> The output folder', type=str, required=True)
    return parser


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_labels = labels[0, 0, 0], labels[0, 0, -1], labels[0, -1, 0], labels[-1, 0, 0]
    background_label = np.argmax(np.bincount(background_labels))

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


# DO THIS ONLINE
# MIN_BOUND = -1000.0
# MAX_BOUND = 400.0
# PIXEL_MEAN = 0.25
def normalize(image, min_bound=-1000, max_bound=400):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


# DO THIS OFFLINE
# pixel_corr = 350
def zero_center(image, pixel_mean=350):
    image = image - pixel_mean
    return image


def resize(img, shape=(50, 50, 20)):
    img = img.transpose(2, 1, 0)
    zoom_factors = [i/float(j) for i, j in zip(shape, img.shape)]
    img = scipy.ndimage.interpolation.zoom(img, zoom=zoom_factors, order=1, mode='nearest')
    return img


# Driver function
def main():
    parser = make_arg_parser()
    args = parser.parse_args()

    # Some constants
    input_folder = args.input
    input_folder = os.path.abspath(input_folder)
    patients = os.listdir(input_folder)
    patients.sort()

    # All patients
    for i, patient in enumerate(patients):
        t0 = time.clock()
        curr_patient = load_scan(os.path.join(input_folder, patient))
        curr_patient_pixels = get_pixels_hu(curr_patient)

        pix_resampled, spacing = resample(curr_patient_pixels, curr_patient, [1, 1, 1])

        del curr_patient
        del curr_patient_pixels

        # TODO: remember to first apply a dilation morphological operation

        pix_resampled = morphology.dilation(pix_resampled)
        # Not sure exactly what that means
        segmented_lungs_fill_masked = segment_lung_mask(pix_resampled, True)
        # plot_3d(segmented_lungs_fill_masked - segmented_lungs, 0)

        pix_resampled = normalize(pix_resampled, min_bound=args.min_bound,
                                     max_bound=args.max_bound)

        pix_resampled = np.multiply(pix_resampled, segmented_lungs_fill_masked)

        print(pix_resampled.shape)
        pix_resampled = resize(pix_resampled, shape=(120, 120, 120))

        pix_resampled = zero_center(pix_resampled, pixel_mean=args.pixel_mean)

        # plot_3d(pix_resampled, os.path.join(args.output, '%s.svg' % patient), threshold=0)

        # pixel_corr = int((args.max_bound - args.min_bound) * args.pixel_mean)  # in this case, 350
        #
        # zero_centered_image = zero_center(segmented_lungs_fill_masked, pixel_corr=pixel_corr)

        save(pix_resampled, os.path.join(args.output, '%s.npz' % patient))

        # Assert all the shapes are the same
        # if i > 0:
        #     assert last_shape == zero_centered_image.shape

        last_shape = pix_resampled.shape
        print(last_shape)

        if args.debug:
            loaded = load(os.path.join(args.output, '%s.npz' % patient))
            assert (loaded == pix_resampled).all()
            break

        print('Files processed: %d' % (i + 1))
        print('Time for file: %.5f' % (time.clock() - t0))

    # DO THIS ONLINE
    # segmented_lungs_fill = normalize(segmented_lungs_fill, min_bound=args.min_bound, max_bound=args.max_bound)

# Used for thread safety
if __name__ == '__main__':
    main()
