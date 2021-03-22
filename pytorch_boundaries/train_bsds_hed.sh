#!/usr/bin/bash

LEARNING_RATE=1e-6
WEIGHT_DECAY=2e-4
LABEL_GAMMA=$1
LABEL_LAMBDA=1.
USE_VAL_FOR_TRAIN=True
BATCH_SIZE=1
NUM_EPOCHS=20
DECAY_STEPS=10000
UPDATE_ITERS=10
MODEL_NAME="vgg16"
OPTIMIZER="sgd"
EXPT_NAME="${2}_bsds_paper_hed_xmax_255_hardlabels_${MODEL_NAME}_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_gamma_${LABEL_GAMMA}_lambda_${LABEL_LAMBDA}_batch_${BATCH_SIZE}_opt_${OPTIMIZER}_update_iters_${UPDATE_ITERS}_nepochs_${NUM_EPOCHS}_decaysteps_${DECAY_STEPS}_use_val_for_train_${USE_VAL_FOR_TRAIN}"
CHECKPOINT=""
DATA_DIR="/mnt/cube/projects/bsds500/HED-BSDS/"
BASE_DIR="models/expt_checkpoints"

echo "Running ${EXPT_NAME}"

python main_bsds.py \
 --learning_rate=${LEARNING_RATE} \
 --weight_decay=${WEIGHT_DECAY} \
 --label_gamma=${LABEL_GAMMA} \
 --label_lambda=${LABEL_LAMBDA} \
 --use_val_for_train=${USE_VAL_FOR_TRAIN} \
 --batch_size=${BATCH_SIZE} \
 --num_epochs=${NUM_EPOCHS} \
 --decay_steps=${DECAY_STEPS} \
 --update_iters=${UPDATE_ITERS} \
 --model_name=${MODEL_NAME} \
 --optimizer=${OPTIMIZER} \
 --expt_name=${EXPT_NAME} \
 --checkpoint=${CHECKPOINT} \
 --data_dir=${DATA_DIR} \
 --base_dir=${BASE_DIR}
