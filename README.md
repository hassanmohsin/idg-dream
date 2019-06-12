# IDG-DREAM challenge
This repository contains code for training machine learning models for the IDG-DREAM challenge

## Setting up and building the models
* Install python (Both python 2 and python 3)
* Run `./setup.sh`. This will create two virtual environments (python2 and python3). Python 2 is required for extracting protein features.
* Run `./prepare_data.sh`. This will extract protein features.
* Run `./train.sh` to train the models. 

The models are trained by leave-on-out method i.e., all the actives for a protein is kept in the test set and the rest are used as the training data. The models are saved in the `models` directory.
