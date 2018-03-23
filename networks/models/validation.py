import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import horovod.tensorflow as hvd

import matplotlib
matplotlib.use('Agg')

import os

def get_hist_bins(data, bins=None, get_error=False, range=(-1.1,1.1)):
    bins = 60 if bins is None else bins
    y, x = np.histogram(data, bins=bins, range=range)
    x = 0.5*(x[1:]+x[:-1])
    if get_error == True:
        y_err = np.sqrt(y)
        return x, y, y_err
    else:
        return x, y


def compute_evaluation_stats(fake, test):
  test_bins, test_hist = get_hist_bins(test)
  fake_bins, fake_hist = get_hist_bins(fake)
  KS_stat, pval = stats.ks_2samp(test_hist, fake_hist)
  return {"KS":pval}


def plot_pixel_histograms(fake, test, dump_path="./", tag=""):
  test_bins, test_hist, test_err = get_hist_bins(test, get_error=True)
  fake_bins, fake_hist, fake_err = get_hist_bins(fake, get_error=True)
  ks_test = stats.ks_2samp(test_hist, fake_hist)[1]

  fig, ax = plt.subplots(figsize=(7,6))
  #plot test
  ax.errorbar(test_bins, test_hist, yerr=test_err, fmt='--ks', \
  label='Test', markersize=7)

  # plot generated
  fake_label = 'GAN-' + tag if tag is not None else "GAN"
  ax.errorbar(fake_bins, fake_hist, yerr=fake_err, fmt='o', \
             label=fake_label, linewidth=2, markersize=6);

  ax.legend(loc="best", fontsize=10)
  ax.set_yscale('log');
  ax.set_xlabel('Pixel Intensity', fontsize=18);
  ax.set_ylabel('Counts (arb. units)', fontsize=18);
  plt.tick_params(axis='both', labelsize=15, length=5)
  plt.tick_params(axis='both', which='minor', length=3)
  # plt.ylim(5e-10, 8*10**7)
  # plt.xlim(-0.3,1.1)
  plt.title('Pixels distribution (KS=%2.3f)'%ks_test, fontsize=16);

  if dump_path is None:
    return None

  if not os.path.exists(dump_path):
    try:
      os.makedirs(dump_path)
    except:
      print("Rank {}: path {} does already exist.".format(hvd.rank(),dump_path))

  plots_dir = os.path.join(dump_path, tag)
  if not os.path.exists(plots_dir):
    try:
      os.makedirs(plots_dir)
    except:
      print("Rank {}: path {} does already exist.".format(hvd.rank(),os.path.join(dump_path,tag)))

  plt.savefig('%s/pixel_intensity.jpg'%plots_dir,bbox_inches='tight', format='jpg')
  plt.savefig('%s/pixel_intensity.pdf'%plots_dir,bbox_inches='tight', format='pdf')


def dump_samples(images, dump_path="./", tag=""):

    # save np arrays
    # np.savez(os.path.join(dump_path, "%s_images.npz"%tag), images)

    # save figures
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8,7.6), 
                           gridspec_kw={'wspace':0.02, 'hspace':0.00})
    idx = np.random.randint(0, images.shape[0], 9)
    for i, a in zip(idx, ax.flatten()):
        a.imshow(images[i])
        a.axis("off")
    fig.suptitle(tag, fontsize=18)
    fig.subplots_adjust(top=0.95)
    fig.savefig('%s/%s.jpg'%(dump_path, tag.replace(" ", "_")),
                bbox_inches='tight', pad_inches=0, format='jpg')
    fig.savefig('%s/%s.eps'%(dump_path, tag.replace(" ", "_")),
                bbox_inches='tight', pad_inches=0, format='eps')

