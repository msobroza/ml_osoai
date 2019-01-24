#!/bin/bash
# You need to modify the dataset path. 
DATA_DIR="./packed_features"

# You can to modify to your own workspace. 
# WORKSPACE=`pwd`
WORKSPACE="./pub_audioset_classification"

BACKEND="pytorch"     # 'pytorch' | 'keras'

MODEL_TYPE="decision_level_single_attention"    # 'decision_level_max_pooling'
                                                # | 'decision_level_average_pooling'
                                                # | 'decision_level_single_attention'
                                                # | 'decision_level_multi_attention'

# Train
CUDA_VISIBLE_DEVICES=1 python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE train

# Calculate averaged statistics. 
python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE get_avg_stats