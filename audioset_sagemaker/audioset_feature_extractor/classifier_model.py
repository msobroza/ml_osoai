import numpy as np
import h5py
import keras
from keras_self_attention import SeqSelfAttention
from keras.models import Model
from keras.layers import (Input, Dense, Concatenate, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Reshape, TimeDistributed, Flatten, LSTM, Bidirectional)
import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
from autopool import AutoPool1D

def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return sample_num, freq_bins

def get_classifier_model(model_type='adaptative_pooling', model_path='weights/md_50000_adap_iters.h5'):


    time_steps = 10
    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units_rnn = 256
    hidden_units = 1024
    drop_rate = 0.5
    batch_size = 500
    
    # Embedded layers
    if model_type=='adaptative_pooling':
        input_layer = Input(shape=(None, freq_bins))
    else:
        input_layer = Input(shape=(time_steps, freq_bins))
    a1 = Dense(hidden_units)(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)

    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)

    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)
    # Pooling layers
    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.

        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(max_pooling, output_shape=pooling_shape)([cla])

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.

        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(
            average_pooling,
            output_shape=pooling_shape)(
            [cla])

    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.

        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        att = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])
    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.

        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        cla1 = Dense(classes_num, activation='sigmoid')(a2)
        att1 = Dense(classes_num, activation='softmax')(a2)
        out1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla1, att1])

        cla2 = Dense(classes_num, activation='sigmoid')(a3)
        att2 = Dense(classes_num, activation='softmax')(a3)
        out2 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla2, att2])

        b1 = Concatenate(axis=-1)([out1, out2])
        b1 = Dense(classes_num)(b1)
        output_layer = Activation('sigmoid')(b1)
    elif model_type == 'feature_level_attention':
        '''Feature level attention.

        [1] To be appear.
        '''
        cla = Dense(hidden_units, activation='linear')(a3)
        att = Dense(hidden_units, activation='sigmoid')(a3)
        b1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

        b1 = BatchNormalization()(b1)
        b1 = Activation(activation='relu')(b1)
        b1 = Dropout(drop_rate)(b1)

        output_layer = Dense(classes_num, activation='sigmoid')(b1)
    elif model_type == 'adaptative_pooling':
        '''
        Adaptive pooling operators for weakly labeled sound event detection
        https://github.com/marl/autopool
        '''

        rnn1 = Bidirectional(LSTM(units=hidden_units_rnn, return_sequences=True))(a3)
        p_dynamic = [TimeDistributed(Dense(1, activation='sigmoid'))(rnn1) for i in range(classes_num)]
        p_static_array = [AutoPool1D(axis=1, kernel_constraint=keras.constraints.non_neg())(p) for p in p_dynamic]
        output = Concatenate()(p_static_array)
        p_dynamic_layer = Concatenate()(p_dynamic)
    else:
        raise Exception("Incorrect model_type!")

        # Build model
    model = Model(inputs=input_layer, outputs=[output, p_dynamic_layer])
    model.summary()
    model.load_weights(model_path, by_name=True)
    return model
