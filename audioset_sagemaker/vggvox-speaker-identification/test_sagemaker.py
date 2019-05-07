import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import boto3

from wav_reader import get_fft_spectrum
import constants as c
import json
import argparse

parser = argparse.ArgumentParser(description='Compares the inference in keras and in tf.')
parser.add_argument("--test_path", default='./samples',
                    help='Path to a directory that contains wav files. Should contain signed 16-bit PCM samples'
                         ' If none is provided, a synthetic sound is used.')
parser.add_argument("--query", default=None,
                    help='Path to the VGGish PCA parameters file.')
parser.add_argument("--endpoint_feature", default='vggvox-features',
                    help='Name of sagemaker endpoint of feature extraction')

FLAGS = parser.parse_args()

endpoint_feat_extract = FLAGS.endpoint_feature
client = boto3.client('runtime.sagemaker', region_name='eu-west-1')


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def get_embedding(wav_file, max_sec):
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    signal = get_fft_spectrum(wav_file, buckets)
    input_signal = signal.reshape(1, *signal.shape, 1).tolist()
    response = client.invoke_endpoint(EndpointName=endpoint_feat_extract, Body=json.dumps(input_signal))
    body = response['Body'].read().decode('utf-8')
    embedding_vox = np.squeeze(np.array(json.loads(body)['outputs']['vgg_features']['floatVal']))
    return embedding_vox


def main():
    print(get_embedding('ph_1.wav', c.MAX_SEC))


if __name__ == '__main__':
    main()
