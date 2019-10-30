#! /bin/bash

DATA=${1}
if [ "$DATA" = "CIFAR10" ]; then
    SWALR=0.01
elif [ "$DATA" = "CIFAR100" ]; then
    SWALR=0.01
elif [ "$DATA" = "DOGFISH" ]; then
    SWALR=0.01
else
    echo "unknown dataset"
fi
MODEL=LogisticLP
SEED=${2}
python3 train.py \
    --dataset ${DATA} \
    --batch_size=128\
    --data_path . \
    --dir block_${MODEL}/${DATA}_${MODEL} \
    --log_name block-${DATA}-${MODEL} \
    --model ${MODEL} \
    --epochs=50 \
    --lr_init=0.05 \
    --swa_start 40 \
    --swa_lr ${SWALR}\
    --wl-weight 8\
    --wl-grad 8 \
    --wd=5e-3 \
    --seed ${SEED} \
    --save_freq 50 \
    --quant-type nearest \
    --small-block Conv \
    --block-dim B;

    #--wl-weight 8 \
    #--wl-acc 8 \
    #--wl-grad 8 \
    #--wl-activate 8 \
    #--wl-error 8 \
