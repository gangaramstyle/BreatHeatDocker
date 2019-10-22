import os
import time

import HeatMatrix as network
import tensorflow as tf
import SODTester as SDT
import SODLoader as SDL
import SOD_Display as SDD
import glob
from pathlib import Path
import numpy as np

sdl= SDL.SODLoader(str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 370, """Batch 1""")
tf.app.flags.DEFINE_integer('batch_size', 370, """Number of images to process in a batch.""")

# Testing parameters
tf.app.flags.DEFINE_string('RunInfo', 'Dice2/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 1, """Which GPU to use""")
tf.app.flags.DEFINE_string('test_files', 'RISK_3', """Testing files""")
tf.app.flags.DEFINE_integer('sleep', 0, """ Time to sleep before starting test""")
tf.app.flags.DEFINE_integer('gifs', 0, """ save gifs or not""")

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_integer('box_dims', 1024, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 256, """the dimensions fed into the network""")
tf.app.flags.DEFINE_integer('net_type', 1, """ 0=Segmentation, 1=classification """)

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 0.5, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-3, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_integer('loss_class', 1, """For classes this and above, apply the above loss factor.""")

# Hyperparameters to control the optimizer
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Define a custom training class
def test():


    # Makes this the default graph where all ops will be added
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        data, iterator = network.inputs(training=False, skip=True)

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        #  Perform the forward pass:
        logits, l2loss = network.forward_pass_unet(data['data'], phase_train=phase_train)

        # Retreive softmax
        softmax = tf.nn.softmax(logits)

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_MAE, best_epoch = 0.25, 0

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as mon_sess:

                # Print run info
                print("*** Validation Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.RunInfo)

                # Initialize the variables
                mon_sess.run(var_init)

                # Initialize iterator
                mon_sess.run(iterator.initializer)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the model
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                else:
                    print ('No checkpoint file found')
                    break

                # Initialize some variables
                step = 0
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)
                sdt = SDT.SODTester(True, False)

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        y_pred, examples = mon_sess.run([softmax, data], feed_dict={phase_train: False})

                        # Increment step
                        step += 1

                except tf.errors.OutOfRangeError:
                    print('Done with Training - Epoch limit reached')

                finally:

                    # TODO: Testing:
                    mask = np.squeeze(examples['label_data'])
                    mask[mask>1] = True
                    heatmap = y_pred[..., 1]
                    blank_heatmap = heatmap * mask
                    # sdd.display_volume(heatmap[30:45], False, cmap='jet')
                    # sdd.display_volume(heatmap[30:45], False, cmap='jet')
                    # sdd.display_volume(blank_heatmap[30:45], True, cmap='jet')
                    for z in range (200, 264):
                        idd = examples['patient'][z]
                        sdd.display_single_image(heatmap[z], False, cmap='jet', title=idd)
                    sdd.display_single_image(heatmap[0], cmap='jet', title=idd)

                    # Testing
                    pred_map = sdt.return_binary_segmentation(y_pred, 0.5, 1, True)
                    print ('Epoch: %s, Best Epoch: %s (%.3f)' %(Epoch, best_epoch, best_MAE))
                    dice, mcc = sdt.calculate_segmentation_metrics(pred_map, examples['label_data'])

                    # Lets save runs that perform well
                    if mcc >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filenames
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, ('Epoch_%s_DICE_%0.3f' % (Epoch, sdt.AUC)))

                        # Save the checkpoint
                        saver.save(mon_sess, checkpoint_file)

                        # Save a new best MAE
                        best_MAE = mcc
                        best_epoch = Epoch

                        if FLAGS.gif:

                            # Delete prior screenshots
                            if tf.gfile.Exists('testing/' + FLAGS.RunInfo + 'Screenshots/'):
                                tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo + 'Screenshots/')
                            tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo + 'Screenshots/')

                            # Plot all images
                            for i in range (FLAGS.batch_size):

                                file = ('testing/' + FLAGS.RunInfo + 'Screenshots/' + 'test_%s.gif' %i)
                                sdt.plot_img_and_mask3D(examples['data'][i], pred_map[i], examples['label_data'][i], file)

                    # Shut down the session
                    mon_sess.close()

            # Print divider
            print('-' * 70)

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                time.sleep(5)

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir+FLAGS.RunInfo + '*')


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(FLAGS.sleep)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    test()

if __name__ == '__main__':
    tf.app.run()