######DOCKER VERSION#######
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
	F/U studies will be no cancer if the pt got CPREV
	CPRV Grp is low risk

RiskStudy:
	(201*2) RiskStudy/HighRisk/Cancer/4/imgxx.dcm (2 views each pt.)
	(Low risk is low risk with no cancer)
	(High Risk / Normal didnt get cancer)
"""

import numpy as np
import tensorflow as tf
import SODKit.SODLoader as SDL
import SODKit.SOD_Display as SDD
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

sdl = SDL(data_root=home_dir)
sdd = SDD()


def pre_process_BRCA(box_dims=1024):
    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, brca_dir)
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number, two different if statements because...

        # /BRCA/G3_Neg_NoMutations/3/Serxx.dcm
        # /BRCA/G3_Neg_MinorMutations/3/Serxx.dcm
        # /BRCA/G1_PosBRCA/NoCancer/Patient 1/L CC/Ser.dcm
        # /BRCA/G1_PosBRCA/Cancer/Patient 1/L CC/Ser.dcm

        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (BRCApos_Cancer_1, BRCAneg_NoMutations_1)
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        if 'G1_PosBRCA' in file:
            patient = 'BRCApos_' + file.split('/')[-4] + '_' + file.split('/')[-3].split(' ')[-1]
            view = patient + '_' + file.split('/')[-2].replace(' ', '')
            if 'NoCancer' in file:
                cancer = 0
            else:
                cancer = 1
        else:
            patient = 'BRCAneg_' + file.split('/')[-3].split('_')[-1] + '_' + file.split('/')[-2]
            view = patient + '_' + file.split('/')[-1].split('.')[0][-1]
            cancer = 0

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        try:
            if 'CC' not in str(header['tags'].ViewPosition): continue
        except:
            continue

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

        # Risk = grp 1
        group = 0

        # Save the data
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 1, file_root='data/BRCA_CC')


def pre_process_RISK(box_dims=1024):
    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, risk_dir)
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number...
        # /RiskStudy/LowRisk/Normal/1/imgxxx.dcm (1+)
        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (RiskHigh_Cancer_1, RiskNl_Normal_1)
        View = unique to that view
        Label = 1 if cancer, 0 if not 
        """

        patient = file.split('/')[-4] + '_' + file.split('/')[-3] + '_' + file.split('/')[-2]
        view = patient + '_' + file.split('/')[-1].split('.')[0][-1]
        # if 'CC' not in view: continue

        if 'Cancer' in file:
            cancer = 1
        else:
            cancer = 0

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        try:
            if 'CC' not in str(header['tags'].ViewPosition): continue
        except:
            continue

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
        image *= mask

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Risk = grp 1
        group = 1

        # Save the data
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s RISK boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/RISK')


def pre_process_CALCS(box_dims=1024):
    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, calc_dir)
    shuffle(filenames)

    # Don't include folder 3 or 4 dicoms
    filenames = [x for x in filenames if '/1/' in x or '/2/' in x]

    # Global variables
    display, counter, data, index, pt = [], [0, 0], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number

        # /Calcs/Eduardo/ADH/Patient 25 YES/1/ser42096img00006.dcm'
        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (CALCSADH_19_YES)
        View = unique to that view (CALCSADH_19_YES_CC)
        Label = 1 if cancer, 0 if not 
        """

        patient = 'CALCS' + file.split('/')[-4] + '_' + file.split('/')[-3].split(' ')[1] + '_' + file.split('/')[-1].split('.')[0][-1]
        view = patient + '_' + ('CC' if '/1/' in file else 'MLO')

        # All of these are technically cancer
        cancer = 1

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        try:
            if 'CC' not in str(header['tags'].ViewPosition): continue
        except:
            continue

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
        image *= mask

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Risk = grp 1
        group = 0

        # Save the data
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s CALCS boxes from %s patients' % (index, pt,))

    # Save the data. Saving with size 2 goes sequentially
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 1, file_root='data/CALCS_CC')


def pre_process_PREV(box_dims=1024, index=0):
    """
    Loads the chemoprevention CC files
    THESE ARE DEIDENTIFIED ON SKYNET!! - Makes the code different
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Chemoprevention/'
    filenames = sdl.retreive_filelist('dcm', True, path)
    shuffle(filenames)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/cprv_all.csv')

    # Include the baseline and follow up chemoprevention studies of the correct breast
    filenames = [x for x in filenames if '1yr_FU' not in x]
    filenames = [x for x in filenames if '#' not in x]

    # Global variables
    display, counter, data, pt = [], [0, 0], {}, 0

    for file in filenames:

        """
        Retreive patient number
        # Chemoprevention/Treated/Patient 2/L CC 5YR/xxx.dcm
        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (CALCSADH_19_YES)
        View = unique to that view (CALCSADH_19_YES_CC)
        Label = 1 if cancer, 0 if not 
        """

        # CC Only
        if 'CC' not in file and 'XCC' not in file: continue

        # Load the Dicom
        try:
            image, accno, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # Re-Identify the file and retreive info
        MRN = header['tags'].PatientID

        try:
            label = lbl_csv[MRN]
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

        # Default to not treated, yes cancer (all high risk)
        treated, cancer = '0', 1

        if '5YR' in file:

            view = 'CPRV5yr_' + MRN + '_' + accno + '_' + file.split('/')[-2].replace(' ', '')[:3]
            # Get cancer and prev status
            if 'Y' in label['Chemoprevention']:
                treated = '1'
                cancer = 0

        else:
            view = 'CPRV0yr_' + MRN + '_' + accno + '_' + file.split('/')[-2].replace(' ', '')[:3]
            cancer = 1
            if 'Y' in label['Chemoprevention']: treated = '1'


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
        try: image *= mask
        except: continue

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Risk = grp 1
        group = 0

        # Save the data
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': MRN, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s Chemoprevention boxes from %s patients' % (index, pt,))

    # Save the data.
    sdl.save_tfrecords(data, 1, file_root='data/CPRV_B5_CC')
    # sdd.display_volume(np.asarray(display, np.float32), True)
    # return data


def pre_process_1YR(box_dims=1024):

    """
    Loads the 1yr followup chemoprevention files
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Chemoprevention/1yr_FU/Processed_CC'
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
        proj = base.split('_')[-1]
        _indexID = base.split('_')[-2]
        MRN = header['tags'].PatientID

        # Set info
        view = 'CPRV1yr_' + MRN + '_' + accno + '_' + proj
        group = '1YR'
        try:
            label = lbl_csv[MRN]
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
            cancer = 0
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

        # Risk = grp 1
        group = 0

        # Save the data
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': MRN, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s Chemoprevention 1yr boxes from %s patients' % (index, pt,), counter)

    # TODO: Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 1, file_root='data/CPRV_1YR_CC')
    # return data


def pre_process_ADJ(box_dims=1024):

    """
    Loads the Adjuvant endocrine therapy patients
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Adjuvant/Processed_CC'
    filenames = sdl.retreive_filelist('dcm', True, path)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/Adj_labels.csv')

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
        proj = base.split('_')[-1]
        MRN = str(header['tags'].PatientID)
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        # Set info
        view = 'ADJ_' + MRN + '_' + accno + '_' + proj
        group = 'ADJ'
        try:
            label = lbl_csv[MRN]
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

        # Get the time since diagnosis date
        DxYr = '20' + label['DxDate'].split('/')[-1]
        TimeSince = 12 * (int(date[:4]) - int(DxYr)) + (int(date[4:6]) - int(label['DxDate'].split('/')[0]))
        TimeSince /= 12
        TimeSince = max(int(round(TimeSince)), 0)

        # # TODO: Check timeSince
        # print ('%s *** %s (%s - %s) M0? %s' %(view, TimeSince, date, label['DxDate'], label['M0']))

        # Get cancer and prev status
        CaSide, cancer = 'R', 0
        if 'left' in label['Side']: CaSide = 'L'
        if label['M0'] == accno and CaSide in proj: cancer = 1

        """
        We have two methods to generate breast masks, they fail on different examples. 
        Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
        So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True, debug=False)

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
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': TimeSince, 'patient': MRN, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1

        # Save after 2500
        if index % 5000 == 0:
            if index < 6000: sdl.save_dict_filetypes(data[0])
            print('Saving after %s patients' % pt)
            sdl.save_tfrecords(data, 1, file_root=('data/test/ADJ%s_CC' % (index // 5000)))
            del data
            data = {}

        del image, mask, labels

    # Done with all patients
    print('Made %s Adjuvant boxes from %s patients' % (index, pt,), counter)

    # TODO: Save the data.
    if data: sdl.save_tfrecords(data, 1, file_root='data/test/ADJ_Fin')
    # return data


def pre_process_SPH(box_dims=1024):
    """
    Loads the data from the school of public health
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'SPH2/'
    filenames = sdl.retreive_filelist('*', True, path)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0, 0], {}, {}, 0, 0
    fail = 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, _, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # Retreive the info
        MRN = os.path.basename(file)
        accno = MRN.replace('.DCM', '').replace('.dcm', '')
        try: proj = header['tags'].ImageLaterality + header['tags'].ViewPosition
        except:
            fail += 1
            continue

        # Set info
        view = accno
        group = 'SPH2'
        label, cancer = 0, 0

        """
        We have two methods to generate breast masks, they fail on different examples. 
        Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
        So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True, debug=False)

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
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': MRN, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1

        del image, mask, labels

    # Done with all patients
    print('Made %s Adjuvant boxes from %s patients' % (index, pt,), counter)
    print (fail, ' failed...')

    # TODO: Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 3, file_root='data/test/SPH2_')
    # return data


def pre_process_VitD(box_dims=1024):
    """
    Loads the data from the Dr. Crew Vitamin D study
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'VitD/'
    filenames = sdl.retreive_filelist('dcm', True, path)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient info
        """

        # Load the Dicom
        try:
            image, _, photo, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # Retreive the info
        MRN = os.path.basename(file)
        accno = MRN
        proj = header['tags'].ImageLaterality + header['tags'].ViewPosition

        # Set info
        view = 'SPH_' + MRN + '_' + proj
        group = 'SPH'
        label, cancer = 0, 0

        """
        We have two methods to generate breast masks, they fail on different examples. 
        Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
        So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True, debug=False)

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
        data[index] = {'data': image.astype(np.float16), 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': MRN, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1

        del image, mask, labels

    # Done with all patients
    print('Made %s Adjuvant boxes from %s patients' % (index, pt,), counter)

    # TODO: Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_tfrecords(data, 3, file_root='data/test/SPH_')
    # return data


def save_all():
    # Run the 1yr
    data = pre_process_1YR()
    index = len(data)

    # Run the others
    data.update(pre_process_PREV(index=index))

    # Save
    sdl.save_dict_filetypes(data[0])
    print ('Saving %s examples into 4 batches' %len(data))
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/PREV')


# Load the protobuf
def load_protobuf(training=True):
    """
        Loads the protocol buffer into a form to send to shuffle. To oversample classes we made some mods...
        Load with parallel interleave -> Prefetch -> Large Shuffle -> Parse labels -> Undersample map -> Flat Map
        -> Prefetch -> Oversample Map -> Flat Map -> Small shuffle -> Prefetch -> Parse images -> Augment -> Prefetch -> Batch
    """

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16,
                                                    segments='label_data', segments_dtype=tf.uint8,
                                                    segments_shape=[FLAGS.box_dims, FLAGS.box_dims])

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Define our undersample and oversample filtering functions
        _filter_fn = lambda x: sdl.undersample_filter(x['group'], actual_dists=[0.75, 0.25], desired_dists=[0.3, .7])
        _undersample_filter = lambda x: dataset.filter(_filter_fn)
        _oversample_filter = lambda x: tf.data.Dataset.from_tensors(x).repeat(
            sdl.oversample_class(x['group'], actual_dists=[0.75, 0.25], desired_dists=[0.3, .7]))

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size // 2)
        dataset = dataset.repeat(20)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Now we have the labels, undersample then oversample.
        dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)

        # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)

    else:
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training:
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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
        data['data'] = tf.cast(data['data'], tf.float32)
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
        data['data'] = tf.cast(data['data'], tf.float32)
        data['data'] = tf.expand_dims(data['data'], -1)
        data['label_data'] = tf.expand_dims(data['label_data'], -1)

        # Reshape, bilinear for labels, cubic for data
        data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims],
                                              tf.compat.v1.image.ResizeMethod.BICUBIC)
        data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims],
                                                    tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

    return data

# pre_process_BRCA()
# pre_process_RISK()
# pre_process_CALCS()
# pre_process_PREV()
# pre_process_1YR()
# pre_process_ADJ()
# pre_process_SPH()
# pre_process_VitD
