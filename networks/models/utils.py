import os
import tensorflow as tf
import numpy as np

def get_data(filename, fformat, compute_stats = False):
    data = np.load(filename, mmap_mode='r')

    #number of samples
    num_samples = data.shape[0]

    #everybody loads everything
    num_samples_per_rank = num_samples
    start = 0
    end = num_samples
    data = data[start:end,:,:]

    if fformat == 'NHWC':
        data = np.expand_dims(data, axis=-1)
    else: # 'NCHW'
        data = np.expand_dims(data, axis=1)

    minval = 0.
    maxval = 1.
    if compute_stats:
        minval = data.min()
        maxval = data.max()

    return data, minval, maxval

def save_checkpoint(sess, saver, tag, checkpoint_dir, counter, step=False):

    model_name = tag + '.model-'
    if step:
        model_name += 'step'
    else:
        model_name += 'epoch'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)

def load_checkpoint(sess, saver, tag, checkpoint_dir, counter=None, step=False):
    print(" [*] Reading checkpoints...")

    if step:
        counter_name = 'step'
    else:
        counter_name = 'epoch'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        if not counter==None:
            ckpt_name_epoch = ckpt_name[:ckpt_name.find(counter_name)] + counter_name + '-%i'%counter
            if os.path.exists(os.path.join(checkpoint_dir, ckpt_name_epoch+'.index')):
                ckpt_name = ckpt_name_epoch
            else:
                print("Checkpoint for ", counter_name , counter_name, "doesn't exist. Using latest checkpoint instead!")

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False
