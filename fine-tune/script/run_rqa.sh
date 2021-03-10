#!/bin/bash

task=$1
bert=$2
run_dir=$3
runs=$4

#. ~/anaconda2/etc/profile.d/conda.sh

#conda activate p3-torch10


if ! [ -z $5 ] ; then
    export CUDA_VISIBLE_DEVICES=$5
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi

DATA_DIR="../data/"$task

for run in `seq 1 1 $runs`
do
    OUTPUT_DIR="../testing_results/"$run_dir/$run

    mkdir -p $OUTPUT_DIR
    if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
        python ../src/run_rqa.py \
            --bert_model $bert --do_train --do_eval \
            --gradient_accumulation_steps 2 \
            --max_seq_length 128 --train_batch_size 8 --learning_rate 1e-5 --num_train_epochs 3 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
    fi

    if ! [ -e $OUTPUT_DIR/"predictions.json" ] ; then 
        python ../src/run_rrc.py \
            --bert_model $bert --do_eval --max_seq_length 128 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/test_log.txt 2>&1

    fi
    if [ -e $OUTPUT_DIR/"predictions.json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
        rm $OUTPUT_DIR/model.pt
    fi
done
