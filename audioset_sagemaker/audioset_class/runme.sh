#!/bin/bash
# You need to modify the dataset path. 
DATA_DIR="./packed_features"

# You can to modify to your own workspace. 
# WORKSPACE=`pwd`
WORKSPACE="./pub_keras_audioset_classification"

BACKEND="keras"     # 'pytorch' | 'keras'

MODEL_TYPE="decision_level_multi_attention_embedding"    # 'decision_level_max_pooling'
                                                # | 'decision_level_average_pooling'
                                                # | 'decision_level_single_attention'
                                                # | 'decision_level_multi_attention'
						# | 'adaptative_pooling'
						# | 'decision_level_multi_attention_embedding' 
EMBEDDINGS_FILEPATH="~/oso/audioset_label_clustering/onto_df.pkl"

# Train
CUDA_VISIBLE_DEVICES=0 python $BACKEND/main.py --data_dir=$DATA_DIR --embeddings=$EMBEDDINGS_FILEPATH --workspace=$WORKSPACE --model_type=$MODEL_TYPE train

# Calculate averaged statistics. 
#python $BACKEND/main.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE get_avg_stats
