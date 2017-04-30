import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt

import os
import cv2

from common import plot_3d

from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn import cross_validation, metrics
import xgboost as xgb
import scipy.ndimage
from skimage import measure
from keras.applications.imagenet_utils import preprocess_input


def get_dicom(path):
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    slice_thickness = 0
    index = 0
    while slice_thickness == 0:
        try:
            slice_thickness = slices[index].SliceLocation - slices[index+1].SliceLocation
        except AttributeError:
            slice_thickness = slices[index].ImagePositionPatient[2] - slices[index+1].ImagePositionPatient[2]
        index += 1
    slice_thickness = np.abs(slice_thickness)
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

    #debugging
    # b8bb02d229361a623a4dc57aa0e5c485

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    # This is breaking occasionally

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1, mode='nearest')

    return image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def normalize(image, min_bound=-1000., max_bound=400.):
    image = remove_background(image)
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def remove_background(image):
    binary_image = np.array(image > -400, dtype=np.int8)
    # binary_image = morphology.closing(binary_image, morphology.ball(2))
    # binary_image = np.array([clear_border(binary_image[i]) for i in range(binary_image.shape[0])])
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    image[background_label == labels] = -1000
    return image


def get_data_id(path):
    dicom = get_dicom(path)
    sample_image = get_pixels_hu(dicom)
    plt.hist(sample_image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    sample_image = resample(sample_image, dicom, [1, 1, 1])
    print(sample_image.shape)
    plot_3d(sample_image, 700)

    sample_image = normalize(sample_image)
    plt.hist(sample_image.flatten(), bins=80, color='c')
    plt.xlabel("Normalized Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # Show some slice in the middle
    plt.imshow(sample_image[80], cmap=plt.cm.gray)
    plt.show()



    # f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))


    batch = []

    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        tmp = np.transpose(tmp, axes=(1, 2, 0)).astype(np.float32)
        tmp = np.expand_dims(tmp, axis=0)
        tmp = preprocess_input(tmp)
        batch.append(np.array(tmp))

        # if cnt < 20:
        #     plots[cnt // 5, cnt % 5].axis('off')
        #     plots[cnt // 5, cnt % 5].imshow(tmp.reshape(224, 224, 3))
        # cnt += 1

    plt.show()
    batch = np.array(batch)
    return batch


def calc_features(path):
    folder = glob.glob(os.path.join(path, '*'))[0]
    batch = get_data_id(folder)


if __name__ == '__main__':
    DATA_DIR = os.path.join('data')
    DICOM_DIR = os.path.join('data', 'stage1')
    DICOM_DIR_2 = os.path.join('data', 'stage2')
    FEATURES_DIR = os.path.join('features')

    calc_features(DICOM_DIR)
