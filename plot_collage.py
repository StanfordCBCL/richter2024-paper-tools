#!/usr/bin/env python

import pdb
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import f_out, f_picture, model_colors, get_geometries

def plot_collage():
    # get geometries and colors
    geos, cats = get_geometries()

    nx = 9
    ny = 8
    assert nx * ny >= len(geos), "choose larger image grid: " + str(len(geos))
    fig, ax = plt.subplots(nx, ny, figsize=(ny * 2, nx * 2.5), dpi=100)
    ig = 0
    for i in range(nx):
        for j in range(ny):
            ax[i, j].axis("off")
            if ig >= len(geos):
                continue

            geo = geos[ig]
            cat = cats[ig]

            impath = os.path.join(f_picture, geo + '.png')
            im = plt.imread(impath)
            ax[i, j].imshow(im)
            ax[i, j].set_title("$\\bf{"+geo.replace("_", "\_") + "}$", fontsize=18)
            #, fontweight="bold", color=model_colors[cat]
            ig += 1
    fpath = os.path.join(f_out, "collage.png")  # .pgf
    # fig.tight_layout(pad=3.0)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_collage()
