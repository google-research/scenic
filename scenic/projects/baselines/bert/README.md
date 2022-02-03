# BERT

JAX implementation of [BERT](https://arxiv.org/abs/1810.04805) that
follows and reproduces the numbers of the
[official TF implmenetation of BERT](https://github.com/tensorflow/models/tree/master/official/nlp/bert).

The code here at this point supports the pretraining and finetuning/fewshot eval
on GLUE.  We hope to find time to also add support for finetuning on SuperGLUE,
SQUAD, and XTREME based on [the code in Tensorflow model](https://github.com/tensorflow/models/tree/master/official/nlp/finetuning).

### Additional Requirements:
The following command will install the required packages for BERT.
```shell
$ pip install -r scenic/projects/baselines/bert/requirements.txt
```

## Process Datasets
The  code here consumes data with the same format as the official
implementation. So to generate the data, you can follow this instruction,
that is also explained in [BERT official repo](https://github.com/tensorflow/models/tree/master/official/nlp/bert#process-datasets):

So to start, you first need to get the preprocessing code:
```shell
$ git clone https://github.com/tensorflow/models.git
```

### Pre-training

To generate pre-training data, you can use the
[`create_pretraining_data` script](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_pretraining_data.py)
(which is essentially branched from [BERT research repo](https://github.com/google-research/bert))
to get the processed pre-training data.

Running the pre-training script requires an input and output directory, as well
as a vocab file.  Note that `max_seq_length` will need to match the sequence
length parameter you specify when you run pre-training.

Example shell script to call create_pretraining_data.py
```shell
$ export WORKING_DIR='local disk or cloud location'
$ export BERT_DIR='local disk or cloud location'
$ python models/official/nlp/data/create_pretraining_data.py \
  --input_file=$WORKING_DIR/input/input.txt \
  --output_file=$WORKING_DIR/output/tf_examples.tfrecord \
  --vocab_file=$BERT_DIR/wwm_uncased_L-24_H-1024_A-16/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

### Fine-tuning
To prepare the fine-tuning data for final model training, use the
[`create_finetuning_data.py` script](https://github.com/tensorflow/models/blob/master/official/nlp/data/create_finetuning_data.py).
Resulting datasets in `tf_record` format and training meta data should be later
passed to training or evaluation scripts. The task-specific arguments are
described in following sections:

#### GLUE

Users can download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
$ export GLUE_DIR=~/glue
$ export BERT_DIR=gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16

$ export TASK_NAME=MNLI
$ export OUTPUT_DIR=gs://some_bucket/datasets
$ python ../data/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}
```


## Pretrained checkpoints
We will release BERT checkpoints that are pretrained using this code and can be
used with no specific modification or weight surgery.


### Acknowledgment
We would like to thank Valerii Likhosherstov and Yi Tay for their contribution
to the BERT implementation in Scenic.
