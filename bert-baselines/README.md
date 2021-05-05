# Huggingface BERT Fine-tuning
This is an implementation based on [Huggingface Transformers](https://github.com/huggingface/transformers) for fine-tuning BERT models on [Arabic Language Understanding Evaluation (ALUE) Benchmark](https://www.alue.org/) tasks, the implementation is an adaptation from the library's [`run_glue.py`](https://github.com/huggingface/transformers/blob/v2.7.0/examples/run_glue.py) script from version `v2.7.0`.

Fine-tuning using this script will re-produce the baselines results reported in [ALUE leaderboard](https://www.alue.org/leaderboard).


# Prerequisites
Install python dependencies:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Code was tested on `Python3.6` and `Ubuntu 18.04`

# Download datasets
Check the instructions in the repository [README.md](https://github.com/Alue-Benchmark/alue_baselines/blob/master/README.md) file.

# Fine-tuning

## Tasks
Choose the task for fine-tuning by exporting the task name as an environment variable:

```
export TASK_NAME=MQ2Q
```

- MQ2Q: [NSURL 2019: Task8 Semantic Question Similarity in Arabic](https://www.kaggle.com/c/nsurl-2019-task8/)
- MDD: [WANLP 2019: Subtask 1: MADAR Travel Domain Dialect Identification](https://sites.google.com/view/madar-shared-task)
- FID: [IDAT@FIRE 2019: Irony Detection in Arabic Tweets](https://www.irit.fr/IDAT2019/)
- SVREG: [SemEval-2018 Task 1: Affect in Tweets: Task V-reg: Detecting Valence or Sentiment Intensity (regression)](https://competitions.codalab.org/competitions/17751)
- SEC: [SemEval-2018 Task 1: Affect in Tweets: Task E-c: Detecting Emotions (multi-label classification)](https://competitions.codalab.org/competitions/17751)
- OOLD: [OSACT4 2020: Offensive Language Detection](http://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/)
- OHSD: [OSACT4 2020: Hate Speech Detection](http://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/)
- XNLI: [The Cross-Lingual NLI Corpus (XNLI)](https://cims.nyu.edu/~sbowman/xnli/)

## Model
Choose the pre-trained model, baselines report results on both `bert-base-multilingual-uncased` and `asafaya/bert-base-arabic`, although any other huggingface pre-trained BERT model would also work.

```
export MODEL_NAME=bert-base-multilingual-uncased
```

## Running Script
Run the script `run_alue.py`:

```
python run_alue.py \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --output_dir results/$TASK_NAME/ \
  --do_train \
  --do_eval \
  --eval_all_checkpoints
```
Submission files will be generated in `results/` folder.

Check `python run_alue.py -h` for more information on hyperparameters and options.
