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
from scipy import stats
from sklearn import metrics
import pathlib

parser = argparse.ArgumentParser(description='Compares the inference in keras and in tf.')
parser.add_argument("--test_path", default=None,
                    help='Path to a directory that contains wav files. Should contain signed 16-bit PCM samples'
                         ' If none is provided, a synthetic sound is used.')
parser.add_argument("--pca_params", default='vggish_pca_params.npz',
                    help='Path to the VGGish PCA parameters file.')
parser.add_argument("--class_description", default='class_labels_indices_fr.csv',
                    help='Path containing the description of each label')
parser.add_argument("--endpoint_feature", default='vggish-features',
                    help='Name of sagemaker endpoint of feature extraction')
parser.add_argument("--endpoint_class", default='attention-classification',
                    help='Name of sagemaker endpoint of the classifier')
parser.add_argument("--tfrecord_file", default=None,
                    help='Path to a TFRecord file where embeddings will be written.')

FLAGS = parser.parse_args()


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats


def get_wav_description_file(file_path):
    current_dir = pathlib.Path(dir_path)
    current_pattern = "*.wav"
    for current_file in current_dir.glob(current_pattern):
        print(current_file)
        print(current_file.split('.')[0])


def get_dict_sounds():
    index_sound_en = dict()
    index_sound_fr = dict()
    with open(FLAGS.class_description, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        c = 0
        for row in spamreader:
            c += 1
            if c == 1:
                continue
            index_sound_en[int(row[0])] = str(row[2])
            index_sound_fr[int(row[0])] = str(row[4])
    return index_sound_en, index_sound_fr


def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128


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
    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
    client = boto3.client('runtime.sagemaker', region_name='eu-west-1')
    data = np.expand_dims(examples_batch, axis=-1).tolist()
    endpoint_feat_extract = FLAGS.endpoint_feature
    endpoint_classifier = FLAGS.endoint_class
    a = datetime.datetime.now()
    response = client.invoke_endpoint(EndpointName=endpoint_feat_extract, Body=json.dumps(data))
    body = response['Body'].read().decode('utf-8')
    embedding_sound = np.array(json.loads(body)['outputs']['vgg_features']['floatVal']).reshape(-1, vggish_params.EMBEDDING_SIZE)
    if len(embedding_sound.shape) == 2:
        postprocessed_batch_keras = pproc.postprocess_single_sample(embedding_sound, num_secs)
    else:
        postprocessed_batch_keras = pproc.postprocess(embedding_sound)
    postprocessed_batch_keras = uint8_to_float32(postprocessed_batch_keras)
    input_class = np.swapaxes(np.swapaxes(postprocessed_batch_keras, 0, 2), 1, 2).tolist()
    response = client.invoke_endpoint(EndpointName=endpoint_classifier, Body=json.dumps(input_class))
    b = datetime.datetime.now()
    c = b - a
    print('Time sagemaker: {}'.format(int(c.microseconds * 0.001)))
    body = response['Body'].read().decode('utf-8')
    output = np.array(json.loads(body)['outputs']['output']['floatVal']).reshape(-1, len(index_sound))
    # Change this line with the real dataset
    calculate_stats(output, outpu)
    if len(output.shape) == 2:
        output = np.mean(output, axis=0)
    indexes_max = output.argsort()[-5:][::-1]
    for i in indexes_max:
        print(index_sound[i])
        print(output[i])


if __name__ == '__main__':
    main()