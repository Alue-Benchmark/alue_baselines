# ALUE BASELINES


<p align="center"><img src="logo/alue.png" height="100" width="100"> <br />

This repo contains the code needed to reproduce the baselines reported in the ALUE [paper](https://camel.abudhabi.nyu.edu/WANLP-2021-Program/14_Paper.pdf). Please also make sure to visit our website at this [link](https://www.alue.org).



Requirements Installation
-------------------------
To rerun the baselines, please ensure that you clone this repo on a suitable Linux server, and then run the following commands in the root directory of the repo.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Data
----
Please ensure that you download the required data for the competition as per the instructions that can be found on the [benchmark website](https://www.alue.org). While we would have liked to provide the participants with the data directly, this would be very difficult in this case due to the perferenes of the original tasks authors, and a number of data privacy regulations, such as the GDPR.

Non-BERT Based Baselines
------------------------
For non-BERT based baselines (i.e. USE, ELMO, ARAVEC, and FastText), please refer to the `notebooks` directory. These baselines are provided via Jupyter Notebooks that provide in detail the exact steps needed to reporduce the results reported in the original paper. Each task is provided in a separate notebook.

BERT Based Baselines
--------------------
