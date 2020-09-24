######DOCKER VERSION#######
"""
Utilities, like loading our other test sets
"""

import numpy as np
import tensorflow as tf
import SODKit.SODLoader as SDL
import SODKit.SOD_Display as SDD
from pathlib import Path
import os
import pydicom as dicom
import shutil

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/'
risk_dir = home_dir + 'RiskStudy/'
short_fu = risk_dir + '1yr_FU/Raw/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'
chemo_dir = home_dir + 'Chemoprevention/'

sdl = SDL.(data_root=home_dir)
sdd = SDD.()


def re_save_1yr(type='CC'):
    """
    Loads the 1 year followup chemoprevention files and resaves the CC and/or MLO DICOM
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Raw/'
    save_path_CC = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Processed_CC/'
    save_path_All = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Processed_MLO_CC/'
    filenames = sdl.retreive_filelist('**', True, path)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            print('DICOM Error: %s' % e)
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            print('Header error: %s' % e)
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        SOD = header['tags'].DistanceSourceToPatient
        SID = header['tags'].DistanceSourceToDetector
        SOD_SID = int(SID) / int(SOD)
        if SOD_SID > 1.25: continue

        # CC Only!!
        if type == 'CC':
            if view != 'CC' and view != 'XCCL': continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        patient = '1YR_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        if type == 'CC':
            savedir = (save_path_CC + accno + '/')
        else:
            savedir = (save_path_All + accno + '/')
        savefile = savedir + view + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        if index % 100 == 0 and index > 1:
            print('\nSaving pt %s of %s to dest: %s_%s\n' % (index, len(filenames), accno, view))

        # Increment counters
        index += 1
        pt += 1

    print('Done with %s images saved' % index)


def re_save_adjuvant(type='CC'):
    """
    Loads the Adjuvant treatment files and resaves the CC and/or MLO DICOM
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    save_path_CC = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Processed_CC/'
    save_path_All = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Processed_MLO_CC/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            print('DICOM Error: %s' % e)
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            print('Header error: %s' % e)
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            continue
        if SOD_SID > 1.25: continue

        # CC Only!!
        if type == 'CC':
            if view != 'CC' and view != 'XCCL': continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        patient = 'ADJ_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        if type == 'CC':
            savedir = (save_path_CC + accno + '/')
        else:
            savedir = (save_path_All + accno + '/')
        savefile = savedir + view + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        if index % 100 == 0 and index > 1:
            print('\nSaving pt %s of %s to dest: %s_%s\n' % (index, len(filenames), accno, view))

        # Increment counters
        index += 1
        pt += 1

    print('Done with %s images saved' % index)


def save_date_adj():
    """
    Saves the date of each adjuvant file into an accno key dict
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Adjuvant/Processed_CC/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    data = {}
    index = 0

    for file in filenames:

        # Load the Dicom
        try:
            header = sdl.load_DICOM_Header(file, multiple=False)
            accno = file.split('/')[-2]
            series = str(header['tags'].SeriesDescription)
        except:
            continue

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            continue
        if SOD_SID > 1.25: continue

        # Get date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        data[accno] = {'date': date, 'desc': series}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'dates.csv')
    print('Done with %s images saved' % len(data))


def save_date_risk():

    """
    Saves the date of each positive RISK file into an accno key dict
    """

    # Load the filenames and randomly shuffle them
    path = risk_dir
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    data = {}
    index = 0

    for file in filenames:

        # Load the Dicom
        try:
            header = sdl.load_DICOM_Header(file, multiple=False)
            accno = header['tags'].StudyID
            series = str(header['tags'].SeriesDescription)
        except:
            continue

        # Get date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        data[accno] = {'date': date, 'desc': series}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'Risk_dates.csv')
    print('Done with %s images saved' % len(data))


def eval_DLs():

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    root_dir = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/'
    missings = sdl.load_CSV_Dict('ACC', root_dir + 'Adjuvant_Missing.csv')
    filenames = sdl.retreive_filelist('*', True, path)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    # Loop through all the "missing" files
    for acc, dic in missings.items():

        # trackers
        foex, fiex = False, False

        # Make sure the folder even exists
        for file in filenames:
            rt = os.path.basename(file)
            if acc == rt:
                foex = True
                break

        if foex != True:
            #print ('%s on MRN %s has no folder' %(acc, dic['MRN']))
            acc = [int(s) for s in acc if s.isdigit()]
            acc = ''.join(map(str, acc))
            # print (acc)
            continue

        # Make sure folder isn't empty
        subfiles = sdl.retreive_filelist('**', True, file)
        if len(subfiles) == 0:
            #print (acc)
            continue

        # Print the study description
        for sfs in subfiles:
            try:
                hdr = sdl.load_DICOM_Header(sfs, False)
                sdesc = hdr['tags'].StudyDescription
                if not sdesc: continue
                # Filter down to outside studies only
                if 'OUTSIDE' not in sdesc: continue
                if 'MR' in sdesc or 'US' in sdesc: continue
                mfc = hdr['tags'].Manufacturer
                if not mfc: continue
                # filter down to non GE outside studies
                if 'Philips' not in mfc and 'SIEMENS' not in mfc: continue
                # Load the Dicom
                try: _ = sdl.load_DICOM_2D(file)
                except Exception as e:
                    print('DICOM Error: %s' % e)
                    continue
                print (acc, ',', mfc)
                #break
            except:
                continue


def check_new(vtype='CC'):

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/ADJ2/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    display, counter, skipped, data, dataf, index, pt = [], [0, 0], [], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            #print('DICOM Error: %s' % e)
            dataf[index] = {'Accno': file.split('/')[-2], 'info': e}
            index += 1
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            #print('Header error: %s' % e)
            dataf[index] = {'Accno': accno, 'info': e}
            index += 1
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                #print('Header error: %s' % e)
                dataf[index] = {'Accno': accno, 'info': e}
                index += 1
                continue

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined:
            dataf[index] = {'Accno': accno, 'info': header['tags'].BodyPartExamined}
            index += 1
            continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            dataf[index] = {'Accno': accno, 'info': 'No SOD_SID'}
            index += 1
            continue
        if SOD_SID > 1.25:
            dataf[index] = {'Accno': accno, 'info': SOD_SID}
            index += 1
            continue

        # CC Only!!
        if vtype == 'CC':
            if view != 'CC' and view != 'XCCL':
                dataf[index] = {'Accno': accno, 'info': view}
                index += 1
                continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        pat = accno + '_' + laterality + view
        data[index] = pat

        if accno not in display: display.append(accno)
        skipped.append(pat)
        print (pat)

        # Increment counters
        index += 1
        pt += 1

    # Un-fail stage
    print ('Removing fake failures')
    dataf2 = dict(dataf)
    for key, index in data.items():
        for keyf, indexf in dataf.items():
            if indexf['Accno'] == index.split('_')[0]:
                if keyf in dataf2: del dataf2[keyf]


    print('Done with %s patients saved. %s failure are now %s' % (len(display), len(dataf), len(dataf2)))
    sdl.save_Dict_CSV(data, 'Saved.csv')
    sdl.save_Dict_CSV(dataf2, 'failed.csv')
    print ('K')


def check_outside():

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    root_dir = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/'
    missings = sdl.load_CSV_Dict('ACC', root_dir + 'Adjuvant_Missing.csv')
    filenames = sdl.retreive_filelist('*', True, path)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    # Loop through all the "missing" files
    for acc, dic in missings.items():

        # trackers
        foex, fiex = False, False

        # Make sure the folder even exists
        for file in filenames:
            rt = os.path.basename(file)
            if acc == rt:
                foex = True
                break

        if foex != True:
            continue

        # Make sure folder isn't empty
        subfiles = sdl.retreive_filelist('**', True, file)
        if len(subfiles) == 0:
            continue

        # Print the study description
        for sfs in subfiles:
            try:
                hdr = sdl.load_DICOM_Header(sfs, False)
                sdesc = hdr['tags'].StudyDescription
                if not sdesc: continue
                # Filter down to outside studies only
                if 'OUTSIDE' not in sdesc: continue
                if 'MR' in sdesc or 'US' in sdesc: continue
                # Load the Dicom
                try: image, accno, _, _, hdr = sdl.load_DICOM_2D(sfs)
                except Exception as e:
                    print('DICOM Error: %s' % e, sfs)
                    continue

                # Now only dealing with outside films with images
                savename = 'data/saved/' + accno + '_' + str(index) + '.jpg'
                savedcm = savename.replace('.jpg', '.dcm')

                # Save the image
                sdl.save_image(image, savename)
                shutil.copyfile(sfs, savedcm)

                # Increment
                index += 1
            except:
                continue


def save_new_ADJ():
    """
    Saves the redownloaded new baseline accession numbers to niice dicoms
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'ADJ2/'
    save_path_CC = home_dir + 'Adj2_Procd/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    display, counter, skipped, data, dataf, index, pt = [], [0, 0], [], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, _, _, header = sdl.load_DICOM_2D(file)
        except Exception as e:
            # print('DICOM Error: %s' % e)
            dataf[index] = {'Accno': file.split('/')[-2], 'info': e}
            index += 1
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            # print('Header error: %s' % e)
            dataf[index] = {'Accno': accno, 'info': e}
            index += 1
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                # print('Header error: %s' % e)
                dataf[index] = {'Accno': accno, 'info': e}
                index += 1
                continue

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined:
            dataf[index] = {'Accno': accno, 'info': header['tags'].BodyPartExamined}
            index += 1
            continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            dataf[index] = {'Accno': accno, 'info': 'No SOD_SID'}
            index += 1
            continue
        if SOD_SID > 1.25:
            dataf[index] = {'Accno': accno, 'info': SOD_SID}
            index += 1
            continue

        # CC Only!!
        if view != 'CC' and view != 'XCCL':
            dataf[index] = {'Accno': accno, 'info': view}
            index += 1
            continue

        # Set info
        patient = 'ADJ_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        savedir = (save_path_CC + accno + '/')
        savefile = savedir + view + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        if index % 100 == 0 and index > 1:
            print('\nSaving pt %s of %s to dest: %s_%s\n' % (index, len(filenames), accno, view))

        # Increment counters
        index += 1
        pt += 1

    print('Done with %s images saved' % index)


def save_ADJ_Outside(box_dims=1024):
    """
        Saves the outside baseline mammograms to protocol buffers
    """

    # Load the filenames and randomly shuffle them
    rawpath = home_dir + 'ADJOutside/Raw/'
    lblpath = home_dir + 'ADJOutside/Picked/'
    filenames = sdl.retreive_filelist('dcm', True, rawpath)
    lblfiles = sdl.retreive_filelist('jpg', True, lblpath)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/Adj_labels.csv')

    # Global variables
    display, counter, skipped, data, dataf, index, pt = [], [0, 0], [], {}, {}, 0, 0

    for file in filenames:

        # Get baseline info
        basename = os.path.basename(file)
        save_index = basename.split('.')[0].split('_')[-1]
        accno = basename.split('.')[0].split('_')[0]
        proj = None

        # Skip patients without labels
        for _file in lblfiles:
            _basename = os.path.basename(_file)
            _save_index = _basename.split('.')[0].split('_')[-2]
            _accno = _basename.split('.')[0].split('_')[0]
            if _save_index == save_index and _accno == accno:
                proj = _basename.split('.')[0].split('_')[-1]
                break

        if not proj: continue

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

        # get the MRN
        MRN = header['tags'].PatientID

        # Set info
        patient = 'ADJ2_' + MRN + '_' + accno
        view = patient + '_' + proj
        group = 'ADJ2'

        # Get the date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)

        if not date: date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].PerformedProcedureStepStartDate)

        # Set info
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
        mask = sdl.zoom_2D(mask.astype(np.int16), shape)
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

        del image, mask, labels

    # Done with all patients
    print('Made %s Adjuvant boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    print('Saving after %s patients' % pt)
    sdl.save_tfrecords(data, 1, file_root=('data/test/ADJ2_Outside_CC'))


def pre_process_ADJ2(box_dims=1024):
    """
    Loads the Adjuvant endocrine therapy patients from the second round of finding baseline studies...
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Adj2_Procd/'
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
        view = 'ADJ2_' + MRN + '_' + accno + '_' + proj
        group = 'ADJ2'
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
        mask = sdl.zoom_2D(mask.astype(np.int16), shape)
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

        del image, mask, labels

    # Done with all patients
    print('Made %s Adjuvant boxes from %s patients' % (index, pt,), counter)

    # Save data
    print('Saving ADJ2 after %s patients' % pt)
    sdl.save_tfrecords(data, 1, file_root=('data/test/ADJ2_CC'))


def save_ADJ_Outside_date(box_dims=1024):
    """
        Saves the outside baseline mammograms to protocol buffers
    """

    # Load the filenames and randomly shuffle them
    rawpath = home_dir + 'ADJOutside/Raw/'
    lblpath = home_dir + 'ADJOutside/Picked/'
    filenames = sdl.retreive_filelist('dcm', True, rawpath)
    lblfiles = sdl.retreive_filelist('jpg', True, lblpath)

    # labels
    lbl_csv = sdl.load_CSV_Dict('MRN', 'data/Adj_labels.csv')

    # Global variables
    display, counter, skipped, data, dataf, index, pt = [], [0, 0], [], {}, {}, 0, 0

    for file in filenames:

        # Get baseline info
        basename = os.path.basename(file)
        save_index = basename.split('.')[0].split('_')[-1]
        accno = basename.split('.')[0].split('_')[0]
        proj = None

        # Skip patients without labels
        for _file in lblfiles:
            _basename = os.path.basename(_file)
            _save_index = _basename.split('.')[0].split('_')[-2]
            _accno = _basename.split('.')[0].split('_')[0]
            if _save_index == save_index and _accno == accno:
                proj = _basename.split('.')[0].split('_')[-1]
                break

        if not proj: continue

        """
        Retreive patient info
        """

        # Load the Dicom
        try:
            image, _, photo, _, header = sdl.load_DICOM_2D(file)
            if photo == 1: image *= -1
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # get the MRN
        MRN = header['tags'].PatientID

        # Set info
        patient = 'OUTSIDE_' + MRN + '_' + accno
        view = patient + '_' + proj

        # Get the date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)

        if not date: date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].PerformedProcedureStepStartDate)

        # Set info
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

        # Get date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        data[accno] = {'date': date, 'desc': view, 'Time_Since_Dx': TimeSince}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'Outside_dates.csv')
    print('Done with %s images saved' % len(data))


def pre_process_ADJ2_date(box_dims=1024):
    """
    Loads the Adjuvant endocrine therapy patients from the second round of finding baseline studies...
    :param box_dims: dimensions of the saved images
    From the reprocessed files which should be all CC views
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Adj2_Procd/'
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
        view = 'ADJ2_' + MRN + '_' + accno + '_' + proj
        group = 'ADJ2'
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

        # Get cancer and prev status
        CaSide, cancer = 'R', 0
        if 'left' in label['Side']: CaSide = 'L'
        if label['M0'] == accno and CaSide in proj: cancer = 1

        data[accno] = {'date': date, 'desc': view, 'Time_Since_Dx': TimeSince}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'ADJ2_dates.csv')
    print('Done with %s images saved' % len(data))


# save_new_ADJ()
# save_ADJ_Outside()
# pre_process_ADJ2()
# save_date_adj()
# save_date_risk()
# eval_DLs()
# check_new()
# check_outside()
save_ADJ_Outside_date()
pre_process_ADJ2_date()
