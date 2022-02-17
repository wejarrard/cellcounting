from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()

def main():
    model = StarDist2D.from_pretrained('2D_versatile_he')

    # for i in 
    example(model, 0, 'data/test/*.tif')



def example(model, i, pathToImage, show_dist=True):

    #
    X = sorted(glob(pathToImage))
    X = list(map(imread,X))

    # Normalize channels
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0,1)   # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    img = normalize(X[i], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(13,10))
    img_show = img if img.ndim==1 else img[...,0]
    coord, points, prob = details['coord'], details['points'], details['prob']
    _draw_polygons(coord, points, prob, show_dist=show_dist)
    plt.subplot(121); plt.imshow(img_show, cmap='gray'); plt.axis('off')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    plt.show()
    


main()