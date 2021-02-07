import h5py
import numpy as np
import pandas as pd

def STEAD_h5(h5):
    return h5py.File(h5, 'r')

def STEAD_csv(data_csv):
   return pd.read_csv(data_csv)
 
