import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

def plot_slice(img, slice=80):
    # Show some slice in the middle
    plt.imshow(img[slice])
    plt.show()


def plot_3d(image, threshold=-100):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    # p = image.transpose(2,1,0)
    p = image

    results = measure.marching_cubes(p, threshold)
    verts = results[0]
    faces = results[1]

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

    plt.savefig('plot3d.png')


def save(arr, pth):
    with open(pth, 'wb+') as fh:
        np.savez_compressed(fh, data=arr)


def load(pth):
    return np.load(pth)['data']


def read_mapping_file(pth):
    return pd.read_csv(pth)

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
