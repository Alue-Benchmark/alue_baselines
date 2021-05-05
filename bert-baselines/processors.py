# Modified version of GLUE processors script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/src/transformers/data/processors/glue.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ALUE processors and helpers """

import logging
import csv
import os

from transformers.file_utils import is_tf_available
from utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def alue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: ALUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = alue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = alue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if example.label is None:
            label = None
        elif output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        elif output_mode == "multilabel":
            label = [int(l) for l in example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %s)" % (example.label, str(label)))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class Mq2qProcessor(DataProcessor):
    """Processor for the Mawdoo3 Question2Question data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/nsurl/q2q_similarity_workshop_v2.1.tsv"), "train")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/q2q/q2q_no_labels_v1.0.tsv"), "test")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/q2q/q2q_no_labels_v1.0.tsv")

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = [("QuestionPairID", "prediction")]
        for (i, line) in enumerate(lines[1:]):
            rows.append((line[0], label_map[preds[i]]))

        self._write_tsv(os.path.join(results_dir, "../q2q.tsv"), rows, delimiter="\t")

    def get_labels(self):
        """See base class."""
        return ["1", "0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = line[1]
                label = line[-1]
            else:
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[1]
                text_b = line[2]
                label = None

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MddProcessor(DataProcessor):
    """Processor for the Madar Dialect Detection data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/madar-1/MADAR-Corpus-26-train.tsv"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/madar-1/MADAR-Corpus-26-test.tsv"), "dev_matched")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../data/madar-1/MADAR-Corpus-26-test.tsv")

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = []
        for (i, line) in enumerate(lines):
            rows.append((label_map[preds[i]],))

        self._write_tsv(os.path.join(results_dir, "../madar.tsv"), rows)

    def get_labels(self):
        """See base class."""
        return ['SAL', 'BEI', 'RIY', 'BAS', 'JER', 'SFX', 'ASW', 'ALX', 'DAM', 'CAI', 'TRI', 'ALE', 'AMM',
                'SAN', 'DOH', 'BAG', 'TUN', 'BEN', 'KHA', 'MUS', 'FES', 'ALG', 'JED', 'MSA', 'RAB', 'MOS']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[-1] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FidProcessor(DataProcessor):
    """Processor for the Fire Iroy Detection data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/idat/IDAT_training_text.csv",
                                                    quotechar="\"", delimiter=","), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/idat/IDAT_test_text.csv",
                                                    quotechar="\"", delimiter=","), "dev_matched")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../data/idat/IDAT_test_text.csv", quotechar="\"", delimiter=",")

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = [("id", "prediction")]
        for (i, line) in enumerate(lines[1:]):
            rows.append((i, label_map[preds[i]]))

        self._write_tsv(os.path.join(results_dir, "../irony.tsv"), rows, delimiter="\t")

    def get_labels(self):
        """See base class."""
        return ["1", "0"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            label = line[2] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SvregProcessor(DataProcessor):
    """Processor for the SemEval V-reg data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/affect-in-tweets/V-reg/2018-Valence-reg-Ar-train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/affect-in-tweets/V-reg/2018-Valence-reg-Ar-dev.txt"), "dev_matched")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/vreg/vreg_no_labels_v1.0.tsv"), "test")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/vreg/vreg_no_labels_v1.0.tsv")

        rows = [("ID", "prediction")]
        for (i, line) in enumerate(lines[1:]):
            rows.append((line[0], preds[i]))

        self._write_tsv(os.path.join(results_dir, "../v_reg.tsv"), rows, delimiter="\t")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            label = line[-1] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SecProcessor(DataProcessor):
    """Processor for the SemEval E-c data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/affect-in-tweets/emotion-c/2018-E-c-Ar-train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/affect-in-tweets/emotion-c/2018-E-c-Ar-dev.txt"), "dev_matched")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/emotion/emotion_no_labels_v1.0.tsv"), "test")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/emotion/emotion_no_labels_v1.0.tsv")

        rows = [("ID", "anger", "anticipation", "disgust", "fear", "joy", "love",
                       "optimism", "pessimism", "sadness", "surprise", "trust")]
        for (i, line) in enumerate(lines[1:]):
            rows.append((line[0],) + tuple([int(pred) for pred in preds[i]]))

        self._write_tsv(os.path.join(results_dir, "../E_c.tsv"), rows, delimiter="\t")

    def get_labels(self):
        """See base class."""
        return ["anger", "anticipation", "disgust", "fear", "joy", "love",
                "optimism", "pessimism", "sadness", "surprise", "trust"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            label = line[-11:] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class OoldProcessor(DataProcessor):
    """Processor for the OSACT4 Offensive Language Detection data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/osact4/OSACT2020-sharedTask-train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/osact4/OSACT2020-sharedTask-dev.txt"), "dev_matched")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/offensive/tweets_v1.0.txt"), "test")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/offensive/tweets_v1.0.txt")

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = []
        for (i, line) in enumerate(lines):
            rows.append((label_map[preds[i]],))

        self._write_tsv(os.path.join(results_dir, "../offensive.tsv"), rows)

    def get_labels(self):
        """See base class."""
        return ["OFF", "NOT_OFF"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class OhsdProcessor(DataProcessor):
    """Processor for the OSACT4 Hate Speech Detection data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/osact4/OSACT2020-sharedTask-train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/osact4/OSACT2020-sharedTask-dev.txt"), "dev_matched")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/offensive/tweets_v1.0.txt"), "test")

    def create_submission(self, results_dir, preds):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/offensive/tweets_v1.0.txt")

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = []
        for (i, line) in enumerate(lines):
            rows.append((label_map[preds[i]],))

        self._write_tsv(os.path.join(results_dir, "../hate.tsv"), rows)

    def get_labels(self):
        """See base class."""
        return ["HS", "NOT_HS"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[2] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class XnliProcessor(DataProcessor):
    """Processor for the The Cross-Lingual NLI Corpus data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/xnli/arabic_train.tsv"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv("../data/xnli/arabic_dev.tsv"), "dev_matched")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv("../private_datasets/diagnostic.tsv"), "test")

    def create_submission(self, results_dir, preds, test=True):
        """Associate predictions with the input IDs to create submission file"""
        lines = self._read_tsv("../private_datasets/diagnostic.tsv" if test else "../data/xnli/arabic_dev.tsv")
        lines = lines[1:]

        label_map = {i: label for i, label in enumerate(self.get_labels())}

        rows = [("pairID", "prediction")]

        for (i, line) in enumerate(lines):
            rows.append((i if test else line[0], label_map[preds[i]]))

        self._write_tsv(os.path.join(results_dir, "../diagnostic.tsv" if test else "../xnli.tsv"), rows, delimiter="\t")

    def get_labels(self):
        """See base class."""
        return ["neutral", "contradiction", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


alue_tasks_num_labels = {
    "mq2q": 2,
    "mdd": 26,
    "fid": 2,
    "svreg": 1,
    "sec": 11,
    "oold": 2,
    "ohsd": 2,
    "xnli": 3,
}


alue_processors = {
    "mq2q": Mq2qProcessor,
    "mdd": MddProcessor,
    "fid": FidProcessor,
    "svreg": SvregProcessor,
    "sec": SecProcessor,
    "oold": OoldProcessor,
    "ohsd": OhsdProcessor,
    "xnli": XnliProcessor,
}


alue_output_modes = {
    "mq2q": "classification",
    "mdd": "classification",
    "fid": "classification",
    "svreg": "regression",
    "sec": "multilabel",
    "oold": "classification",
    "ohsd": "classification",
    "xnli": "classification",
}
