# ARRU phase picker: Attention-Recurrent-Residual-U-Net-for-earthquake-detection
We're working on a more stable model on processing continuous seismograms as well as an useful repository! 
Here are just the simple scripts for model training and prediction using STandford Earthquake Dataset (STEAD) dataset. 

# Equipments
tensorflow-gpu >=2.0.0 
tensorflow-addons 0.11.2 
(any version equipped with 'tensorflow_addons.optimizers.RectifiedAdam' module is fine)

# Script piplines 
Below describes the workflow from data generation, model training, making predictions, to model evaluation. 

1. Prepare the seismic recordings from STEAD data : `P01_make_stream_STEAD.py`

This script simply generates sac files as well as TFRecord in length of 20 seconds. This would require the STEAD dataset (https://github.com/smousavi05/STEAD), please download and place the 'merge.hdf5' (you could retreive this entire STEAD dataset here: https://mega.nz/folder/HNwm0SLY#h70tuXK2tpiQJAaPq72FFQ) file in the directory './data'. You can change the variable 'csv_type' line 22 with ['train', 'test', 'val'] to generate dataset we used in our study according to the list stored in './data/partition_csv'. Noted that you have to make './data/partition_csv/train_STEAD.csv' on your own according to uploaded file './partition_csv/test_STEAD.csv' from complete information of STEAD dataset.
-----
output directory: (1) './data/sac_data_STEAD_20s' (2) './input_TFRecord_STEAD_20s'

2. 

# Reference
Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen, Ruey‐Juin Rau; ARRU Phase Picker: Attention Recurrent‐Residual U‐Net for Picking Seismic P‐ and S‐Phase Arrivals. Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200382
