######DOCKER VERSION#######
import os
import time

import HeatMatrix as network
import tensorflow as tf
import SODKit.SODTester as SDT
import SODKit.SODLoader as SDL
import SODKit.SOD_Display as SDD
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import numpy.ma as ma


_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
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

# convert the pre_process code to be dataset agnostic
def pre_process(f_dir, output_key):
    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    sdl = SDL(data_root=f_dir)
    sdd = SDD()
    box_dims=1024

    # Load the filenames
    filenames = sdl.retreive_filelist('dcm', True, f_dir)

    # Global variables
    record_index, file_index = 0, 0
    # Failure trackers
    f_dicom, f_cc, f_laterality, f_mask_generate, f_mask_apply = 0, 0, 0, 0, 0
    data = {}

    for file_index, file in enumerate(filenames):

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            f_dicom += 1
            continue

        try:
            if 'CC' not in str(header['tags'].ViewPosition): continue
        except:
            f_cc += 1
            continue

        laterality = ''
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except:
                f_laterality += 1
                print(header)
                input('...')
                continue

        view = laterality + '_' + header['tags'].ViewPosition

        """
                We have two methods to generate breast masks, they fail on different examples. 
                Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
                So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True)

        # Some masks just won't play ball
        mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])
        if mask_idx > 0.8 or mask_idx < 0.1:
            print('Failed to generate mask... ', view)
            f_mask_generate += 1
            continue

        # Multiply image by mask to make background 0
        try:
            mask = mask.astype(image.dtype)
            image *= mask
        except:
            f_mask_apply += 1
            print('Failed to apply mask... ', view, image.shape, image.dtype, mask.shape, mask.dtype)
            continue

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[record_index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'view': view, 'accno': accno}

        # Increment counters
        record_index += 1
        del image, mask, labels

    # Done with all patients
    #print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 1, file_root=f"data/{output_key}/{output_key}")
    return(record_index)

# Define a custom training class
def inference(output_key):
    sdd = SDD()


    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Load the images and labels.
        iterator = network.inputs(training=False, skip=True)
        data = iterator.get_next()

        # Define input shape
        data['data'] = tf.reshape(data['data'], [FLAGS.batch_size, FLAGS.network_dims, FLAGS.network_dims])

        #  Perform the forward pass:
        # TODO: Double check w/ simi on the unet stuff
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

        while True:

            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            with tf.Session(config=config) as mon_sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

                # Initialize the variables
                mon_sess.run([var_init, iterator.initializer])

                # Finalize the graph to detect memory leaks!
                mon_sess.graph.finalize()

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the model
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                else:
                    print('No checkpoint file found')
                    break

                # Initialize some variables
                step = 0
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)
                #TODO: Make sure the flags are correct
                sdt = SDT(True, False)

                # Dictionary of arrays merging function
                def _merge_dicts(dict1={}, dict2={}):
                    ret_dict = {}
                    for key, index in dict1.items(): ret_dict[key] = np.concatenate([dict1[key], dict2[key]])
                    return ret_dict

                try:
                    while step < max_steps:

                        # Load some metrics for testing
                        _softmax_map_, _data_ = mon_sess.run([softmax_map, data], feed_dict={phase_train: False})
                        # Combine cases
                        del _data_['data']
                        if step == 0:
                            _softmax_map = np.copy(_softmax_map_)
                            _data = dict(_data_)
                        else:
                            _softmax_map = np.concatenate([_softmax_map, _softmax_map_])
                            _data = _merge_dicts(_data, _data_)

                        # Increment step
                        step += 1

                except Exception as e:
                    print(e)

                finally:

                    # Get the background mask
                    mask = np.squeeze(_data['label_data'] > 0).astype(np.bool)

                    # Get the group 1 heatmap and group 2 heatmap
                    heatmap_high = _softmax_map[..., 1]
                    heatmap_low = _softmax_map[..., 0]

                    # Make the data array to save
                    save_data, high_scores, low_scores, display = {}, [], [], []
                    high_std, low_std = [], []
                    for z in range(FLAGS.epoch_size):
                        # TODO: Adjust the appropriate keys
                        # Generate the dictionary
                        save_data[z] = {
                            'Accno': _data['accno'][z].decode('utf-8'),
                            'Image_Info': _data['view'][z].decode('utf-8'),
                            'Cancer Score': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).mean(),
                            'Benign Score': ma.masked_array(heatmap_low[z].flatten(), mask=~mask[z].flatten()).mean(),
                            'Max Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).max(),
                            'Min Value': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).min(),
                            'Standard Deviation': ma.masked_array(heatmap_high[z].flatten(),
                                                                  mask=~mask[z].flatten()).std(),
                            'Variance': ma.masked_array(heatmap_high[z].flatten(), mask=~mask[z].flatten()).var(),
                        }

                        
                        """ 
                        Make some corner pixels max and min for display purposes
                        Good Display Runs: Unet_Fixed2 epoch 60, and Initial_Dice epoch 149
                        """
                        image = np.copy(heatmap_high[z]) * mask[z]
                        # max, min = 0.9, 0.2
                        max, min = np.max(heatmap_high[z]), np.min(heatmap_high[z])
                        image[0, 0] = max
                        image[255, 255] = min
                        image = np.clip(image, min, max)
                        # sdd.display_single_image(image, True, title=save_data[z]['Image_Info'], cmap='jet')
                        save_file = 'imgs/' + save_data[z]['Image_Info'] + '.png'
                        save_file = save_file.replace('PREV', '')
                        if 'CC' not in save_file: continue
                        # sdd.save_image(image, save_file)
                        # plt.imsave(save_file, image, cmap='jet')

                        # Generate image to append to display
                        # display.append(np.copy(heatmap_high[z]) * mask[z])
                        display.append(image)

                    # Save the data array
                    #High, Low = float(np.mean(np.asarray(high_scores))), float(np.mean(np.asarray(low_scores)))
                    #hstd, lstd = float(np.mean(np.asarray(high_std))), float(np.mean(np.asarray(low_std)))
                    #diff = High - Low
                    #print('Epoch: %s, Diff: %.3f, AVG High: %.3f (%.3f), AVG Low: %.3f (%.3f)' % (
                    #    Epoch, diff, High, hstd, Low, lstd))
                    print(save_data)
                    sdt.save_dic_csv(save_data, f"{output_key}_{FLAGS.RunInfo.replace('/', '')}.csv", index_name='ID')

                    # Now save the vizualizations
                    # sdl.save_gif_volume(np.asarray(display), ('testing/' + FLAGS.RunInfo + '/E_%s_Viz.gif' % Epoch), scale=0.5)
                    sdd.display_volume(display, True, cmap='jet')

                    del heatmap_high, heatmap_low, mask, _data, _softmax_map

                    # Shut down the session
                    mon_sess.close()
            break

home_dir = '/app/data/raw'
output_key = 'pprocessed'
Path(f"/app/data/{output_key}/").mkdir(parents=True, exist_ok=True)
tf.app.flags.DEFINE_string('data_dir', f"/app/data/{output_key}/", """Path to the data directory.""")

record_num = pre_process(home_dir, output_key)

#TODO: figure out epoch_size and batch_size intelligently
tf.app.flags.DEFINE_integer('epoch_size', record_num, """SPH2 - 131""")
tf.app.flags.DEFINE_integer('batch_size', record_num, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")

tf.app.flags.DEFINE_string('RunInfo', 'Combined2/', """Unique file name for this training run""")
inference(output_key)
FLAGS['RunInfo'].value = 'UNet_Fixed2/'
inference(output_key)
