
# --------------------------------------------------------
#                DATA FOLDERS
# --------------------------------------------------------

DATA_FOLDER = 'ieeg/'
# DATA_FOLDER='/home/shlomo/dev/data-sets/ieeg/'

DATA_FOLDER_IN = DATA_FOLDER + "input/"
DATA_FOLDER_OUT = DATA_FOLDER + "output/"

BAD_FILES_TRAIN=DATA_FOLDER_IN + "bad_files_train.csv"
BAD_FILES_TEST=DATA_FOLDER_IN + "bad_files_test.csv"

import pandas
colnames = ['file']

#BAD_FILES_TRAIN_LIST=np.loadtxt(BAD_FILES_TRAIN,dtype=str,skiprows=1,usecols=(1,))

TRAIN_PREFIX_ALL = "train_all"
TRAIN_DATA_FOLDER_IN_ALL= DATA_FOLDER_IN + "/" + TRAIN_PREFIX_ALL + "/"

TRAIN_PREFIX_1 = "train_1"
TRAIN_DATA_FOLDER_IN_1 = DATA_FOLDER_IN + "/" + TRAIN_PREFIX_1 + "/"

TRAIN_PREFIX_2 = "train_2"
TRAIN_DATA_FOLDER_IN_2 = DATA_FOLDER_IN + "/" + TRAIN_PREFIX_2 + "/"

TRAIN_PREFIX_3 = "train_3"
TRAIN_DATA_FOLDER_IN_3 = DATA_FOLDER_IN + "/" + TRAIN_PREFIX_3 + "/"

TRAIN_FEAT_BASE= DATA_FOLDER_OUT + "/feat_train/"


TEST_PREFIX_ALL = "test_all"
TEST_DATA_FOLDER_IN_ALL = DATA_FOLDER_IN + "/" + TEST_PREFIX_ALL + "/"

TEST_PREFIX_1 = "test_1"
TEST_DATA_FOLDER_IN_1 = DATA_FOLDER_IN + "/" + TEST_PREFIX_1 + "/"

TEST_PREFIX_2 = "test_2"
TEST_DATA_FOLDER_IN_2 = DATA_FOLDER_IN + "/" + TEST_PREFIX_2 + "/"

TEST_PREFIX_3 = "test_3"
TEST_DATA_FOLDER_IN_3 = DATA_FOLDER_IN + "/" + TEST_PREFIX_3 + "/"

TEST_FEAT_BASE= DATA_FOLDER_OUT + "/feat_test/"

#hdf5_encoded_train = DATA_FOLDER_OUT + TRAIN_PREFIX + '-.hdf5'
#saved_results_pkl_train = DATA_FOLDER_OUT + TRAIN_PREFIX + '-lm.pkl'

#hdf5_encoded_test = DATA_FOLDER_OUT + TEST_PREFIX + '-.hdf5'
#saved_results_pkl_test = DATA_FOLDER_OUT + TEST_PREFIX + '-lm.pkl'


# DEFINE THE RESPONSE VARIABLE FOR THIS DATA SET
singleResponseVariable = 'result'

# LIBFM_PATH = '/home/shlomo/dev/new-db/libfm/bin/libFM'
#LIBFM_PATH = '/Volumes/3pt-enc/db/Dropbox/dev/python/libfm/'

INTERICTAL = 0
PREICTAL = 1
# --------------------------------------------------------
#                DATA FOLDERS
# --------------------------------------------------------

        
## Assigning data types to attributes:
from datetime import datetime
import numpy


