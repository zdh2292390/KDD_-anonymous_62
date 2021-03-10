data_dir=data/Electronics
out=output/ck
task_name=ahm
#train_size=16
eval_size=64
norm=1
warmup=0
decay=0.0
lr=1e-5

for epoch in 10
do
  for bsz in 32
  do
    echo epoch = ${epoch}, lr = ${lr}, batch_size = ${bsz}
    python pretrain_AHM.py \
      --model_type=bert \
      --data_dir=${data_dir} \
      --model_name_or_path=bert-base-uncased \
      --task_name=${task_name} \
      --do_train --do_eval \
      --output_dir=${out}_ep${epoch}_lr${lr}_bs${bsz} \
      --learning_rate=${lr} \
      --num_train_epochs=${epoch} \
      --max_grad_norm=${norm} \
      --warmup_steps=${warmup} \
      --weight_decay=${decay} \
      --train_batch_size=${bsz} \
      --eval_batch_size=${eval_size}
    exit
  done
done

task_name=cea
for epoch in 10
do
  for bsz in 32
  do
    echo epoch = ${epoch}, lr = ${lr}, batch_size = ${bsz}
    python pretrain_CEA.py \
      --model_type=bert \
      --data_dir=${data_dir} \
      --model_name_or_path=bert-base-uncased \
      --task_name=${task_name} \
      --do_train --do_eval \
      --output_dir=${out}_ep${epoch}_lr${lr}_bs${bsz} \
      --learning_rate=${lr} \
      --num_train_epochs=${epoch} \
      --max_grad_norm=${norm} \
      --warmup_steps=${warmup} \
      --weight_decay=${decay} \
      --train_batch_size=${bsz} \
      --eval_batch_size=${eval_size}
    exit
  done
done


