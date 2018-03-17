import os
import tensorflow as tf

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

        restore_latest=False
        if counter:
            ckpt_name_epoch = ckpt_name[:ckpt_name.find(counter_name)] + counter_name + '-%i'%counter
            if os.path.exists(os.path.join(checkpoint_dir, ckpt_name_epoch+'.index')):
                ckpt_name = ckpt_name_epoch
            else:
                print("Checkpoint for ", counter_name , counter_name, "doesn't exist. Using latest checkpoint instead!")
                restore_latest=True
        else:
            restore_latest=True
    
        #look for latest checkpoint
        if restore_latest:
            #get list of checkpoints
            checkpoints = [x.replace(".index","") for x in os.listdir(checkpoint_dir) if x.startswith("model.ckpt") and x.endswith(".index")]
            checkpoints = sorted([(int(x.split("-")[1]),x) for x in checkpoints], key=lambda tup: tup[0])
            ckpt_name = checkpoints[-1][1]

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False
