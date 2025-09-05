#!/bin/bash
## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"

nvidia-smi
export CUDA_VISIBLE_DEVICES=0
source ~/proto/bin/activate

# use this line to run the main.py file with a specified config file
# example: python3 run.py --config_path="path/to/file"
<< Experiments :
Experiments
############################# General Env Parameters ############################
SAVE_DIR="/results"
CONFIG_YML="src/configs/config.yml"
CHECKPOINT_PATH="$SAVE_DIR/<checkpoint>.pth"
<< Basic :
experimenting the prototype video-based Resnet2p1D
Basic
#################################################################################
########### Get the scatterplot and Regression performance on Test ##############
#################################################################################
TEST_TYPE="TEST"
DATA_TYPE="all" # all, single,2clip, 3clip
B=25
P=4
FRAMES=36
python ./main.py --config_path=$CONFIG_YML --run_name=$NAME --save_dir=$SAVE_DIR \
        --eval_only=True --eval_data_type=$TEST_TYPE \
        --model.checkpoint_path=$CHECKPOINT_PATH \
        --data.test_type=$DATA_TYPE --wandb_mode="online" 


#################################################################################
################## Extract the Features and Prototype Vectors ###################
#################################################################################
NAME="features_00"
OUT_FILE="$OUT_DIR/featureExtract_00.out"
DATA_TYPE="all" 
python extract_features.py --config_path=$CONFIG_YML --save_dir=$SAVE \
      --run_name=$NAME --data.test_type=$DATA_TYPE \
      --model.checkpoint_path=$CHECKPOINT_PATH  --wandb_mode="disabled" 


#################################################################################
######################### Visualise Features on pca ########################
################################################################################# 
NAME="umap_00"
OUT_FILE="$OUT_DIR/umap_00.out"
python plotpca.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR \
      --model.checkpoint_path=$CHECKPOINT_PATH --wandb_mode="disabled" 


#################################################################################
############### Get the Local Explaination on Given Test Subset #################
#################################################################################
CSV_FILE="explain.csv"
NAME="Explainer_00"
python explain.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME \
      --abstain_class=False --wandb_mode="disabled" --run_name="ExtractFeatures" \
      --data.csv_file=$CSV_FILE --data.label_scheme_name=$LABLE_SCHEMA \
      --explain_locally=True --eval_data_type=$TEST_TYPE --data.test_type=$DATA_TYPE \
      --model.checkpoint_path=$CHECKPOINT_PATH

#################################################################################
echo "Done: $(date +%F-%R:%S)"

