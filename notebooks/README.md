# Non-BERT based Baselines
This directory contains two sub-directories. The first is `USE` which contains notebooks for the reproduction of the Universal Sentence Encoder baslines, per task. The second is `elmo_aravec_fasttext`, which as the name suggests has the notebooks for the reporduction of the ELMO, ARAVEC, and FastText baselines. Each notebook in either subdirectories has a self explanatory name that links to the original task name of concern.

Please make sure that the paths to the datasets in each notebook is correct. The paths provided by us are recommended, and meant to easy the reproduction of all the baselines, BERT-based and otherwise. However, feel free to place them differently, so long as you reflect that on the notebooks.

Installing Pre-trained Models
-----------------------------
We provide a bash file (i.e. `install.sh`) to download the needed pre-trained models for the notebooks to run correctly. To run it please use the following command on a bash terminal:
```
bash install.sh
```
Please note that the models are big in size, and each can take a significant amount of time to download and install in the right directory. If for whatever reason, one of the files needs to be downloaded, you can run the file again, but please make sure you comment out the files that don't need to be redownloaded.
