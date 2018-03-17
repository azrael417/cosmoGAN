import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def sample_tfrecords_to_numpy(tfrecords_filenames, img_size, sess_config, n_samples=1000, normalization=None):

  def decode_record(x):
    parsed_example = tf.parse_single_example(x,
        features = {
            "data_raw": tf.FixedLenFeature([],tf.string)
        }
    )

    example = tf.decode_raw(parsed_example['data_raw'],tf.float32)
    example = tf.reshape(example,[img_size, img_size])
    return example
         
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x: decode_record(x))  # Parse the record into tensors.
  dataset = dataset.repeat(1)  # Repeat the input indefinitely.
  dataset = dataset.batch(n_samples)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  # Initialize `iterator` with training data.
  with tf.Session(config=sess_config) as sess:
      sess.run(iterator.initializer, 
              feed_dict={filenames: tfrecords_filenames})
      images = sess.run(next_element)
      sess.close()

  if normalization is not None:
    pix_min, pix_max = normalization
    images = -1+ 2 * (images - pix_min) / (pix_max - pix_min)

  return images
        

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
  ks_test = compute_evaluation_stats(fake, test)['KS']

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

  if dump_path is not None:
    plots_dir = "%s/%s" % (dump_path, tag)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig('%s/pixel_intensity.jpg'%plots_dir,bbox_inches='tight', format='jpg')
    plt.savefig('%s/pixel_intensity.pdf'%plots_dir,bbox_inches='tight', format='pdf')

