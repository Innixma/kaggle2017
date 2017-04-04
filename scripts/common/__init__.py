import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_slice(img, slice=80):
    # Show some slice in the middle
    plt.imshow(img[slice])
    plt.show()


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


def save(arr, pth):
    with open(pth, 'wb+') as fh:
        np.savez_compressed(fh, data=arr)

def load(pth):
    return np.load(pth)['data']
