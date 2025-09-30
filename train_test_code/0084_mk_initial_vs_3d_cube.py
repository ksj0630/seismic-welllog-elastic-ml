import numpy as np
import pandas as pd
from turtle import shape
import tensorflow as tf
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.models import Model, load_model, save_model
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
import glob
import sys
import re
from tqdm import tqdm


###########################
# Load_Data
###########################
# n1 = 4500
# n2 = 597
# n3 = 401

n1 = 904
n2 = 213
n3 = 152

# Initial P-wave velocity CUBE
fin3 = open("../16.for_MK_3d_cube_data/un2_cut_velp_904.bin","rb")
velp = np.fromfile(fin3,dtype='float32')
velp = velp.reshape(-1,n1)

init_vs_cube = velp / 1.732

fout1 = open('../17.MK_3D_cube/est_cube_initial_Vs.bin', 'wb')
est_cube = np.float32(init_vs_cube)
est_cube.tofile(fout1)
fout1.close()

