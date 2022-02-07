# ARRU phase picker: Attention-Recurrent-Residual-U-Net-for-earthquake-detection
We're working on a more stable model on processing continuous seismograms as well as an useful repository! <br/>
Here are just the simple scripts for model training and prediction using STandford Earthquake Dataset (STEAD) dataset. 

https://user-images.githubusercontent.com/30610646/120765835-327a6300-c54c-11eb-99b2-6ea4bf6b1c94.mp4


# Script piplines 
Below describes the workflow from data generation, model training, making predictions, to model evaluation. 

1. **Prepare the seismic recordings from STEAD data** : **`P01_make_stream_STEAD.py`**<br/>
This script simply generates sac files as well as TFRecord in length of 20 seconds. This would require the [STEAD](https://github.com/smousavi05/STEAD) dataset , please download and place the file 'merge.hdf5' (you could retreive this entire [STEAD](https://mega.nz/folder/HNwm0SLY#h70tuXK2tpiQJAaPq72FFQ) dataset here in the directory '`./data`'. <br/>
You can change the variable '`csv_type`' in line 22 with [`train`, `test`, `val`] to generate dataset we used in our study according to the files stored in '`./data/partition_csv`'. Noted that you have to make '`./data/partition_csv/train_STEAD.csv`' by yourselves according to uploaded lists of STEAD dataset for data partition, if needed. <br/><br/>
Output directory: **(1) './data/sac_data_STEAD_20s' (2) './data/input_TFRecord_STEAD_20s'**

2. **Model training**: <br/>
In this repository we provide two pretrained models,  <br/>**`./pretrained_model/paper_model_ARRU_20s`** and **`./pretrained_model/multitask_ARRU_20s`** <br/> <br/>
The former works as seismic phase picker described in our paper, and the latter one provides an additional mask associating the P and S arrivals, which could also be treated as the earthquake event detector. Both of these models were trained with local earthquake events that the maximum separation between P and S arrival is 12 seconds. <br/> <br/>
**`./P02_train_codes/P01_Unet_train_gpus_STEAD.py`**<br/>
**`./P02_train_codes/P01_Unet_train_detect_gpus_STEAD.py`**<br/>

3. **Making predictions on STEAD dataset**<br/>

4. **Model evaluation** <br/>

5. Quick example of making predictions using pretrained model <br/>
```$ python quick_ex.py```

# Binder link
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection/HEAD)
# Reference
Wu‐Yu Liao, En‐Jui Lee, Dawei Mu, Po Chen, Ruey‐Juin Rau; ARRU Phase Picker: Attention Recurrent‐Residual U‐Net for Picking Seismic P‐ and S‐Phase Arrivals. Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200382
