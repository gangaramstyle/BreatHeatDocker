"""
Does our loading and preprocessing of files to a protobuf

Who are the patients who have cancer in our dataset: (1885 / 1824  no cancer views, 356 / 356 (+407 Ca+) cancer views)

BRCA:
	(37*4 or 146) BRCA/G1_PosBRCA/Cancer/Patient 2/R CC/xxx.dcm (4 views each patient)

Calcs:
    File 1 and File 2 contain CC and MLO views of the affected breast - These pts have cancer though
    The /new data does not contain full field mammograms - dammit man!
	Invasive
	Microinvasion
	(ADH and DCIS is not cancer)

Chemoprevention:
	(All just high risk ADH, LCIS and DCIS, no cancers)

RiskStudy:
	(201*2) RiskStudy/HighRisk/Cancer/4/imgxx.dcm (2 views each pt.)
	(Low risk is low risk with no cancer)
	(High Risk / Normal didnt get cancer)
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import os

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/'

risk_dir = home_dir + 'RiskStudy/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'
chemo_dir = home_dir + 'Chemoprevention/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

def pre_process_PREV(box_dims=1024):
    """
    Loads the chemoprevention CC files
    THESE ARE DEIDENTIFIED ON SKYNET!!
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Chemoprevention/'
    filenames = sdl.retreive_filelist('dcm', True, path)
    shuffle(filenames)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/cprv_all.csv')

    # Include the baseline and follow up studies
    filenames = [x for x in filenames if '#' not in x]

    # Global variables
    display, counter, data, index, pt = [], [0, 0], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        # Chemoprevention/Treated/Patient 2/L CC 5YR/xxx.dcm
        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (CALCSADH_19_YES)
        View = unique to that view (CALCSADH_19_YES_CC)
        Label = 1 if cancer, 0 if not 
        """

        # Load and resize image
        try:
            image, _, _, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
        except:
            print("Failed to load: ", file)
            continue

        # Re-Identify the file and retreive info
        info = sdl.ReID_DICOMs('@@Igadbrain316', file)
        accno = info['Accession']
        patient = info['MRN']
        if '5YR' in file:
            group = '5YR'
        else:
            group = 'BASE'
        view = 'CPRV_' + accno + '_' + file.split('/')[-2].replace(' ', '')[:3]
        try:
            label = lbl_csv[patient]
        except:
            try:
                break_sig = False
                for mr, dic in lbl_csv.items():
                    if break_sig: break
                    for key, val in dic.items():
                        if val == accno:
                            label = dic
                            break_sig = True
                            break
            except Exception as e:
                print('No Label: ', e)
                continue

        # CC Only
        if 'CC' not in view and 'XCC' not in view: continue

        # Get cancer and prev status
        if 'Y' in label['Chemoprevention']:
            treated = '1'
        else:
            treated = '0'
        cancer = 1

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
            continue

        # Multiply image by mask to make background 0
        try:
            image *= mask
        except:
            continue

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': treated, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s Chemoprevention boxes from %s patients' % (index, pt,))

    # Save the data.
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/CPRV_B5_CC')


def pre_process_1YR(box_dims=1024):
    """
    Loads the 1yr followup chemoprevention files
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Reprocessed_1k/'
    filenames = sdl.retreive_filelist('dcm', True, path)
    shuffle(filenames)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/cprv_all.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # Retreive the info
        base, folder = os.path.basename(file).split('.')[0], os.path.dirname(file)
        proj = base.split('_')[-1] + '1YR'
        _indexID = base.split('_')[-2]
        patient = header['tags'].PatientID

        # Set info
        view = 'CPRV_' + accno + '_' + proj
        group = '1YR'
        try:
            label = lbl_csv[patient]
        except:
            try:
                break_sig = False
                for mr, dic in lbl_csv.items():
                    if break_sig: break
                    for key, val in dic.items():
                        if val == accno:
                            label = dic
                            break_sig = True
                            break
            except Exception as e:
                print('No Label: ', e)
                continue

        # Only work on the 1 year followups
        if label['Acc_1yr'] != accno: continue

        # Get cancer and prev status
        if 'Y' in label['Chemoprevention']:
            treated = '1'
        else:
            treated = '0'
        cancer = 1

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
            continue

        # Multiply image by mask to make background 0
        try:
            image *= mask
        except:
            print('Failed to apply mask... ', view, image.shape, image.dtype, mask.shape, mask.dtype)

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': treated, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # TODO: Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/CPRV_1YR_CC')

# Load the protobuf
def load_protobuf(training=True):
    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Define filenames
    filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
    print('******** Loading Files: ', filenames)
    dataset = tf.data.TFRecordDataset(filenames)

    # Repeat input indefinitely
    dataset = dataset.repeat()

    # Shuffle the entire dataset then create a batch
    if training: dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)

    # Load the tfrecords into the dataset with the first map call
    _records_call = lambda dataset: \
        sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float32,
                           segments='label_data', segments_dtype=tf.uint8,
                           segments_shape=[FLAGS.box_dims, FLAGS.box_dims])

    # Parse the record into tensors
    dataset = dataset.map(_records_call, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Fuse the mapping and batching
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):

        # Map the data set
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset and drop remainder. Can try batch before map if map is small
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    # cache and Prefetch
    dataset = dataset.prefetch(buffer_size=FLAGS.batch_size)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):

    self._distords = distords

  def __call__(self, data):

    if self._distords:  # Training

        # Expand dims by 1
        data['data'] = tf.expand_dims(data['data'], -1)
        data['label_data'] = tf.expand_dims(data['label_data'], -1)

        # Reshape, bilinear for labels, cubic for data
        data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims],
                                              tf.compat.v1.image.ResizeMethod.BICUBIC)
        data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims],
                                                    tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Random rotate
        angle = tf.random_uniform([], -0.45, 0.45)
        data['data'] = tf.contrib.image.rotate(data['data'], angle, interpolation='BILINEAR')
        data['label_data'] = tf.contrib.image.rotate(data['label_data'], angle, interpolation='NEAREST')

        # Random shear:
        rand = []
        for z in range(4):
            rand.append(tf.random_uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32))
        data['data'] = tf.contrib.image.transform(data['data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0],
                                                  interpolation='BILINEAR')
        data['label_data'] = tf.contrib.image.transform(data['label_data'],
                                                        [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0],
                                                        interpolation='NEAREST')

        # Randomly flip
        def flip(mode=None):

            if mode == 1:
                img, lbl = tf.image.flip_up_down(data['data']), tf.image.flip_up_down(data['label_data'])
            elif mode == 2:
                img, lbl = tf.image.flip_left_right(data['data']), tf.image.flip_left_right(data['label_data'])
            else:
                img, lbl = data['data'], data['label_data']
            return img, lbl

        # Maxval is not included in the range
        data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random.uniform([], 0, 2, dtype=tf.int32)) > 0,
                                                   lambda: flip(1), lambda: flip(0))
        data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random.uniform([], 0, 2, dtype=tf.int32)) > 0,
                                                   lambda: flip(2), lambda: flip(0))

        # Random contrast and brightness
        data['data'] = tf.image.random_brightness(data['data'], max_delta=2)
        data['data'] = tf.image.random_contrast(data['data'], lower=0.975, upper=1.025)

        # Random gaussian noise
        T_noise = tf.random.uniform([], 0, 0.1)
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)
        data['data'] = tf.add(data['data'], tf.cast(noise, tf.float32))


    else: # Testing

        # Expand dims by 1
        data['data'] = tf.expand_dims(data['data'], -1)
        data['label_data'] = tf.expand_dims(data['label_data'], -1)

        # Reshape, bilinear for labels, cubic for data
        data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims],
                                              tf.compat.v1.image.ResizeMethod.BICUBIC)
        data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims],
                                                    tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

    return data


# pre_process_1YR(1024)
pre_process_PREV(1024)
