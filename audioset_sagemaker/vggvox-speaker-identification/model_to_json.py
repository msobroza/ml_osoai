import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c


def main():
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()
	model.compile(loss='mse', optimizer='adam')
	# serialize model to JSON
	model_json = model.to_json()
	with open("model_vggvox.json", "w") as json_file:
    		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model_vggvox.h5")
	print("Saved model to disk")


if __name__ == '__main__':
	main()
