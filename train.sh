#!/bin/sh
## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

nvidia-smi
####### general env variables
source ~/proto/bin/activate

CONFIG_YML="src/configs/config.yml"
SEED=200
NAME="test"
SAVE_DIR="/$NAME"
SAMPLER="EF"
SCHEME="ef_2class"
DATASET_PATH="/echonet"
EPOCHS=30
PROTOTYPE_SHAPE=(20,256,1,1,1)
python ./main.py --config_path=$CONFIG_YML --run_name=$NAME --save_dir=$SAVE_DIR \
        --model.prototype_shape=$PROTOTYPE_SHAPE \
        --abstain_class=False --model.num_classes=1 --train.num_train_epochs=$EPOCHS --train.seed=$SEED \
        --data.dataset_path=$DATASET_PATH --data.sampler=$SAMPLER --data.label_scheme_name=$SCHEME 
