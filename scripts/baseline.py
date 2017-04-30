import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt

import os
import cv2

from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn import cross_validation, metrics
import xgboost as xgb
import scipy.ndimage
from skimage import measure
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from common import shuffle_weights


def get_extractor():
    model = ResNet50(include_top=False, weights='imagenet')
    if SHUFFLE:
        shuffle_weights(model)
        model.save('models/shuffled_resnet.h5')
    return model


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
    sample_image = resample(sample_image, dicom, [1, 1, 1])
    sample_image = normalize(sample_image)

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
    net = get_extractor()
    for folder in glob.glob(os.path.join(path, '*')):
        base = os.path.basename(folder)
        if not os.path.exists(os.path.join(FEATURES_DIR, '%s.npy' % base)):
            batch = get_data_id(folder)
            feats = []
            for i in range(batch.shape[0]):
                print(batch[0].shape)
                feats.append(net.predict(batch[i]))
            feats = np.array(feats)
            print(feats.shape)
            np.save(os.path.join(FEATURES_DIR, '%s.npy' % base), feats)


def train_xgboost():
    df = pd.read_csv(os.path.join(DATA_DIR, 'stage1_labels.csv'))
    print(df.head())

    mask = np.array([True if os.path.exists(os.path.join(FEATURES_DIR, '%s.npy' % str(id))) else False for id in df['id'].tolist()])

    df = df.ix[mask]

    x = []
    for i, id in enumerate(df['id'].tolist()):
        x.append(np.median(np.load(os.path.join(FEATURES_DIR, '%s.npy' % str(id))), axis=0))
        if i % 15 == 0:
            print(i)
    x = np.array(x)

    y = df['cancer'].as_matrix()

    x = x.reshape((x.shape[0], x.shape[-1]))


    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.10)
    eval_set = [(trn_x, trn_y)]

    clf = xgb.XGBRegressor(max_depth=20,
        n_estimators=10000,
        min_child_weight=20,
        learning_rate=0.05,
        nthread=8,
        subsample=0.80,
        colsample_bytree=0.80,
        seed=3200)

    clf.fit(trn_x, trn_y, eval_set=eval_set, verbose=True, eval_metric=['error', 'logloss'], early_stopping_rounds=50)

    return clf


def make_submit():
    clf = train_xgboost()

    df = pd.read_csv(os.path.join(DATA_DIR, 'stage1_sample_submission.csv'))

    mask = np.array(
        [True if os.path.exists(os.path.join(FEATURES_DIR, '%s.npy' % str(id))) else False for id in df['id'].tolist()])

    df = df.ix[mask]

    x = np.array([np.median(np.load(os.path.join(FEATURES_DIR, '%s.npy' % str(id))), axis=0) for id in df['id'].tolist()])
    x = x.reshape((x.shape[0], x.shape[-1]))

    pred = clf.predict(x)

    df['cancer'] = pred
    df.to_csv('subm1_random.csv', index=False)
    print(df.head())
    return clf


def calc_log_loss():
    df_pred = pd.read_csv('subm1.csv')

    df_truth = pd.read_csv(os.path.join(DATA_DIR, 'stage1_solution.csv'))

    df_truth.sort_values(['id'])
    df_pred.sort_values(['id'])

    pred = df_pred['cancer'].values
    truth = df_truth['cancer'][df_truth['id'].isin(df_pred['id'])].values

    print(metrics.log_loss(truth, pred))
    print(metrics.log_loss(truth, np.array(truth < 2, dtype=float)))
    print(metrics.log_loss(truth, np.array(truth < 0, dtype=float)))
    print(metrics.log_loss(truth, np.repeat(.5, truth.shape[0])))
    print(confusion_matrix(pred > .5, truth))
    print(confusion_matrix(truth < 2, truth))
    print(confusion_matrix(truth < 0, truth))


def plot_roc_curve():
    from sklearn.metrics import roc_curve, auc
    df_pred = pd.read_csv('subm1.csv')

    df_truth = pd.read_csv(os.path.join(DATA_DIR, 'stage1_solution.csv'))

    df_truth.sort_values(['id'])
    df_pred.sort_values(['id'])

    pred = df_pred['cancer'].values
    truth = df_truth['cancer'][df_truth['id'].isin(df_pred['id'])].values

    fpr, tpr, _ = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    df_pred = pd.read_csv('subm1_random.csv')

    df_truth = pd.read_csv(os.path.join(DATA_DIR, 'stage1_solution.csv'))

    df_truth.sort_values(['id'])
    df_pred.sort_values(['id'])

    pred = df_pred['cancer'].values
    truth = df_truth['cancer'][df_truth['id'].isin(df_pred['id'])].values

    fpr_r, tpr_r, _ = roc_curve(truth, pred)
    roc_auc_r = auc(fpr_r, tpr_r)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr_r, tpr_r, color='darkred',
             lw=lw, label='ROC curve random (area = %0.2f)' % roc_auc_r)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_training_loss(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    # plot log lossm
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('XGBoost Log Loss')
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()

if __name__ == '__main__':
    DATA_DIR = os.path.join('data')
    DICOM_DIR = os.path.join('data', 'stage1')
    DICOM_DIR_2 = os.path.join('data', 'stage2')
    SHUFFLE = True
    if SHUFFLE:
        FEATURES_DIR = os.path.join('features_shuffled')
    else:
        FEATURES_DIR = os.path.join('features')

    # calc_features(DICOM_DIR)
    # calc_features(DICOM_DIR_2)
    # clf = make_submit()
    # plot_training_loss(clf)
    calc_log_loss()
    plot_roc_curve()
