#!/usr/bin/env python
import argparse
import os
import numpy as np
import dicom
import scipy.ndimage

DESCRIPTION = """
Explain the script here
"""


def make_arg_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-i', '--input', help='<PATH> The input folder', type=str, required=True)
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

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


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
        # All patients
        for i, patient in enumerate(patients):
            curr_patient = load_scan(os.path.join(input_folder, patient))
            curr_patient_pixels = get_pixels_hu(curr_patient)

            pix_resampled, spacing = resample(curr_patient_pixels, curr_patient, [1, 1, 1])

            print(np.mean(curr_patient_pixels))
            print(np.mean(pix_resampled))



# Used for thread safety
if __name__ == '__main__':
    main()
