# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

import numpy as np
from scipy.io import wavfile
import six
import csv
import vggish_input
import vggish_params
import vggish_postprocess
import datetime
import numpy as np
import argparse
import csv
import sys
import json
import boto3
import numpy as np
import io

parser = argparse.ArgumentParser(description='Compares the inference in keras and in tf.')
parser.add_argument("--wav_file", default=None,
                    help='Path to a wav file. Should contain signed 16-bit PCM samples'
                         ' If none is provided, a synthetic sound is used.')
parser.add_argument("--pca_params", default='vggish_pca_params.npz',
                    help='Path to the VGGish PCA parameters file.')
parser.add_argument("--tfrecord_file", default=None,
                    help='Path to a TFRecord file where embeddings will be written.')

FLAGS = parser.parse_args()


def get_dict_sounds():
    index_sound = dict()
    with open('class_labels_indices.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        c = 0
        for row in spamreader:
            c += 1
            if c == 1:
                continue
            index_sound[int(row[0])] = str(row[2])
    return index_sound


def main():
    index_sound = get_dict_sounds()
    num_secs = 10
    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.
    if FLAGS.wav_file:
        wav_file = FLAGS.wav_file
        print(wav_file)
    else:
        # Write a WAV of a sine wav into an in-memory file object.
        num_secs = 5
        freq = 1000
        sr = 44100
        t = np.linspace(0, num_secs, int(num_secs * sr))
        x = np.sin(2 * np.pi * freq * t)
        # Convert to signed 16-bit samples.
        samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
        wav_file = six.BytesIO()
        wavfile.write(wav_file, sr, samples)
        wav_file.seek(0)
    examples_batch = vggish_input.wavfile_to_examples(wav_file)

    # print(examples_batch)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
    client = boto3.client('runtime.sagemaker', region_name='eu-west-1')
    data = np.expand_dims(examples_batch, axis=-1).tolist()
    endpoint_feat_extract = 'sagemaker-tensorflow-2019-02-08-11-49-31-577'
    endpoint_classifier = 'sagemaker-tensorflow-2019-02-08-12-18-50-613'
    a = datetime.datetime.now()
    response = client.invoke_endpoint(EndpointName=endpoint_feat_extract, Body=json.dumps(data))
    body = response['Body'].read().decode('utf-8')
    embedding_sound = np.array(json.loads(body)['outputs']['vgg_features']['floatVal']).reshape(-1, vggish_params.EMBEDDING_SIZE)
    if len(embedding_sound.shape) == 2:
        postprocessed_batch_keras = pproc.postprocess_single_sample(embedding_sound, num_secs)
    else:
        postprocessed_batch_keras = pproc.postprocess(embedding_sound)
    input_class = np.swapaxes(np.swapaxes(postprocessed_batch_keras, 0, 2), 1, 2).tolist()
    response = client.invoke_endpoint(EndpointName=endpoint_classifier, Body=json.dumps(input_class))
    b = datetime.datetime.now()
    c = b - a
    print('Time sagemaker: {}'.format(int(c.microseconds * 0.001)))
    body = response['Body'].read().decode('utf-8')
    output = np.array(json.loads(body)['outputs']['vgg_features']['floatVal']).reshape(-1, len(index_sound))
    if len(output.shape) == 2:
        output = np.mean(output, axis=0)
    indexes_max = output.argsort()[-5:][::-1]
    for i in indexes_max:
        print(index_sound[i])
        print(output[i])


if __name__ == '__main__':
    main()
