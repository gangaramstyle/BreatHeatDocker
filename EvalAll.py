######DOCKER VERSION#######
import time

from . import HeatMatrix as network
import tensorflow as tf
import SODKit.SODTester as SDT
import SODKit.SODLoader as SDL
import SODKit.SOD_Display as SDD
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.ma as ma

sdl = SDL(str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/')
sdd = SDD()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# >5k example lesions total
tf.app.flags.DEFINE_integer('epoch_size', 14000, """1yr 1895, Other 1941""")
tf.app.flags.DEFINE_integer('batch_size', 175, """1yr 379, Other 647/3""")

# Testing parameters. NC e14 ufixed2 e40, comb2 #2,
tf.app.flags.DEFINE_string('RunInfo', 'Combined2/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")
tf.app.flags.DEFINE_integer('sleep', 0, """ Time to sleep before starting test""")
tf.app.flags.DEFINE_integer('gifs', 0, """ save gifs or not""")

# Define some of the immutable variables
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('data_dir', 'data/test/', """Path to the data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_integer('box_dims', 1024, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 256, """the dimensions fed into the network""")

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
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        #  Perform the forward pass:
        logits, _ = network.forward_pass_unet(data['data'], phase_train=phase_train)

        # Retreive softmax_map
        softmax_map = tf.nn.softmax(logits)

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

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Init SDT
        sdt = SDT(True, False)

        # Run once for all the saved checkpoints
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)
        for checkpoint in ckpt.all_model_checkpoint_paths:

            with tf.Session(config=config) as mon_sess:

                # Print run info
                print("*** Validation Run %s on GPU %s ****" % (FLAGS.RunInfo, FLAGS.GPU))

                # Initialize the variables
                mon_sess.run([var_init, iterator.initializer])

                # Restore the model
                saver.restore(mon_sess, checkpoint)
                Epoch = checkpoint.split('/')[-1].split('_')[-1]

                # Initialize some variables
                step, index, save_data = 0, 0, {}
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        _softmax_map, _data = mon_sess.run([softmax_map, data], feed_dict={phase_train: False})

                        # Increment step
                        step += 1

                        # Update save data
                        save_data.update(stats_adj(_softmax_map, _data, index))
                        index += FLAGS.batch_size

                except Exception as e:
                    print('Training Stopped: ', e)

                finally:

                    save_data = stats_adj_fin(save_data, Epoch)
                    sdt.save_dic_csv(save_data, ('testing/' + FLAGS.RunInfo + '/E_%s_Data.csv' % Epoch), index_name='ID')

                    del _data, _softmax_map, save_data

                    # Shut down the session
                    mon_sess.close()

        plt.show()


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(FLAGS.sleep)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    test()


def stats_prev(_softmax_map, _data, Epoch):
    # Generate a background mask
    mask = np.squeeze(_data['label_data'] > 0).astype(np.bool)

    # Get the group 1 heatmap and group 2 heatmap
    heatmap_high = _softmax_map[..., 1]
    heatmap_low = _softmax_map[..., 0]

    # # Make the data array to save
    save_data, low_scores, high_scores, display = {}, [], [], []
    low_std, high_std = [], []
    for z in range(FLAGS.batch_size):

        # Generate the dictionary
        save_data[z] = {
            'Accno': _data['accno'][z].decode('utf-8'),
            'Cancer Label': int(_data['cancer'][z]),
            # 'Treated': int(_data['group'][z].decode('utf-8')),
            'Image_Info': _data['view'][z].decode('utf-8'),
            'Cancer Score': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).mean(),
            'Benign Score': ma.masked_array(heatmap_low[z].flatten(), mask=~mask[z].flatten()).mean(),
            'Max Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).max(),
            'Min Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).min(),
            'Standard Deviation': ma.masked_array(heatmap_high[z].flatten(),
                                                  mask=~mask[z].flatten()).std(),
            'Variance': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).var(),
        }

        # Append the scores
        if save_data[z]['Cancer Label'] == 1:
            high_scores.append(save_data[z]['Cancer Score'])
        else:
            low_scores.append(save_data[z]['Cancer Score'])
        if save_data[z]['Cancer Label'] == 1:
            high_std.append(save_data[z]['Standard Deviation'])
        else:
            low_std.append(save_data[z]['Standard Deviation'])

        # TODO: Make some corner pixels max and min for display purposes
        # Try: Tightening range, CV scaling
        image = np.copy(heatmap_high[z]) * mask[z]
        # max, min = 0.9, 0.2
        max, min = np.max(heatmap_high[z]), np.min(heatmap_high[z])
        image[0, 0] = max
        image[255, 255] = min
        image = np.clip(image, min, max)
        display.append(image)

        # Save the data array
    High, Low = float(np.mean(np.asarray(high_scores))), float(np.mean(np.asarray(low_scores)))
    hstd, lstd = float(np.mean(np.asarray(high_std))), float(np.mean(np.asarray(low_std)))
    diff = High - Low
    print('Epoch: %s, Diff: %.3f, AVG High: %.3f (%.3f), AVG Low: %.3f (%.3f)' % (
        Epoch, diff, High, hstd, Low, lstd))

    return save_data


def stats_risk(_softmax_map, _data, Epoch):
    # Generate a background mask
    mask = np.squeeze(_data['label_data'] > 0).astype(np.bool)

    # Get the group 1 heatmap and group 2 heatmap
    heatmap_high = _softmax_map[..., 1]
    heatmap_low = _softmax_map[..., 0]

    # Make the data array to save
    save_data, high_scores, low_scores, display = {}, [], [], []
    high_std, low_std = [], []
    for z in range(FLAGS.batch_size):

        # Generate the dictionary
        save_data[z] = {
            'Accno': _data['accno'][z].decode('utf-8'),
            'Cancer Label': int(_data['cancer'][z]),
            'Image_Info': _data['view'][z].decode('utf-8'),
            'Cancer Score': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).mean(),
            'Benign Score': ma.masked_array(heatmap_low[z].flatten(), mask=~mask[z].flatten()).mean(),
            'Max Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).max(),
            'Min Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).min(),
            'Standard Deviation': ma.masked_array(heatmap_high[z].flatten(),
                                                  mask=~mask[z].flatten()).std(),
            'Variance': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).var(),
        }

        # Append the scores
        if save_data[z]['Cancer Label'] == 1:
            high_scores.append(save_data[z]['Cancer Score'])
        else:
            low_scores.append(save_data[z]['Cancer Score'])
        if save_data[z]['Cancer Label'] == 1:
            high_std.append(save_data[z]['Standard Deviation'])
        else:
            low_std.append(save_data[z]['Standard Deviation'])

        # TODO: Make some corner pixels max and min for display purposes
        # Try: Tightening range, CV scaling
        image = np.copy(heatmap_high[z]) * mask[z]
        # max, min = 0.9, 0.2
        max, min = np.max(heatmap_high[z]), np.min(heatmap_high[z])
        image[0, 0] = max
        image[255, 255] = min
        image = np.clip(image, min, max)
        # sdd.display_single_image(image, True, title=save_data[z]['Image_Info'], cmap='jet')
        display.append(image)

        # Generate image to append to display
        # display.append(np.copy(heatmap_high[z]) * mask[z])

    # Save the data array
    High, Low = float(np.mean(np.asarray(high_scores))), float(np.mean(np.asarray(low_scores)))
    hstd, lstd = float(np.mean(np.asarray(high_std))), float(np.mean(np.asarray(low_std)))
    diff = High - Low
    print('Epoch: %s, Diff: %.3f, AVG High: %.3f (%.3f), AVG Low: %.3f (%.3f)' % (
        Epoch, diff, High, hstd, Low, lstd))

    return save_data


def stats_adj(_softmax_map, _data, index):
    # Generate a background mask
    mask = np.squeeze(_data['label_data'] > 0).astype(np.bool)

    # Get the group 1 heatmap and group 2 heatmap
    heatmap_high = _softmax_map[..., 1]

    # Make the data array to save
    save_data = {}
    for z in range(FLAGS.batch_size):
        # Generate the dictionary
        save_data[index + z] = {
            'Accno': _data['accno'][z].decode('utf-8'),
            'Time Since Dx': int(_data['group'][z]),
            'Image_Info': _data['view'][z].decode('utf-8'),
            'Risk Score': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).mean(),
        }

    return save_data


def stats_adj_fin(save_data={}, Epoch=0):
    # Make the data array to save
    low_scores, high_scores, display = [], [], []

    for z in range(FLAGS.epoch_size):

        # Append the scores. Append multiple if later scans
        if save_data[z]['Time Since Dx'] != 0:
            low_scores.extend([save_data[z]['Risk Score'] for i in range(save_data[z]['Time Since Dx'])])
        else:
            high_scores.append(save_data[z]['Risk Score'])

    # Save the data array
    High, Low = float(np.mean(np.asarray(high_scores))), float(np.mean(np.asarray(low_scores)))
    diff = High - Low
    print('Epoch: %s, Diff: %.3f, AVG High: %.3f, AVG Low: %.3f' % (Epoch, diff, High, Low))

    return save_data


if __name__ == '__main__':
    tf.app.run()
