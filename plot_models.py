import json
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style, rcParams
import multiprocessing
from tempfile import TemporaryDirectory
import seaborn as sns

import numpy as np
import pandas as pd
from rich import box, print
from rich.table import Table
from scipy.stats import pearsonr
from svsuperestimator.reader import *
from svsuperestimator.reader import utils as readutils
from svsuperestimator.tasks import taskutils
import pysvzerod
from svsuperestimator.main import run_from_config
import shutil

import matplotlib.patheffects as path_effects

import utils

models_img_path = "/Volumes/richter/final_data/multi_fidelity_calibration/models"

this_file_dir = os.path.abspath(os.path.dirname(__file__))

style.use(os.path.join(this_file_dir, "matplotlibrc"))

width = rcParams["figure.figsize"][0]

fig, axs = plt.subplots(1, 3, figsize=[width, width*0.35], sharey=True)

import matplotlib.image as mpimg

for i, model_name in enumerate(["0104_0001", "0140_2001", "0080_0001"]):
    img = mpimg.imread(models_img_path + f"/{model_name}.png")
    imgplot = axs[i].imshow(img)
    
    axs[i].set_title(model_name)
    axs[i].axis('off')
    fig.subplots_adjust(bottom=0.0, left=0.02, right=0.98, top=0.9, wspace=0.0)

    plt.savefig(models_img_path + "/models_plot.png")