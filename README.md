# SAS

## Install
The code is based on Matterplot3D Simulator, please follow [R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop) to setup the environment. Beside the default env, SAS needs additional libraries:
* lmdb
* scipy
* tslearn

Please run the following line to install the libraries.
```
pip3 install lmdb scipy tslearn
```

**************************************************************

# Data preparation
The ResNet image features of R2R dataset should be placed like:
```
${PROJECT_ROOT}/
|-- img_features
|   |-- ResNet-152-imagenet.tsv
```
The paraphrase data of meteor metric can be downloaded from [here](https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/data/paraphrase-en.gz). It should be put into the following directory:
```
${PROJECT_ROOT}/
|-- r2r_src
|   |-- eval_utils
|      |-- meteor
|         |-- data
|            |-- paraphrase-en.gz
```
The processed detection data can be downloaded from [here](https://drive.google.com/file/d/1NBfsPwee3Xs5gva41nqMiPYKoL0TZmQp/view?usp=sharing). The feature data should be placed in:
```
${PROJECT_ROOT}/
|-- r2r_src
|   |-- detect_feat_genome_by_view.pkl
```

***************************************************************

## Training

Run the following line to start training:

```
bash ./run/speaker_sas.bash [GPU_id]
```

*******************************************************************

# Acknowledgement
Our code is based on the following repository. We thank the authors for releasing their codes. 

- [R2R-EnvDrop](https://github.com/airsplay/R2R-EnvDrop)
