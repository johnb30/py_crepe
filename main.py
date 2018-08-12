"""
Run on GPU: Install tensorflow-gpu (v1.6), then do: python main.py
"""

from __future__ import print_function
from __future__ import division
import json
import py_crepe
import numpy as np
import data_helpers
np.random.seed(123)  # for reproducibility

# set parameters:

subset = None

# Whether to save model parameters
save = False
model_name_path = 'params/crepe_model.json'
model_weights_path = 'params/crepe_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 1014

# Model params
# Filters for conv layers
nb_filter = 256
# Number of units in the dense layer
dense_outputs = 1024
# Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
# Number of units in the final output layer. Number of classes.
cat_output = 4

# Compile/fit params
batch_size = 80
nb_epoch = 20

print('Loading data...')
# Expect x to be a list of sentences. Y to be index of the categories.
(xt, yt), (x_test, y_test) = data_helpers.load_ag_data()

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, alphabet = data_helpers.create_vocab_set()

print('Build model...')
model = py_crepe.create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                              nb_filter, cat_output)
# Encode data
xt = data_helpers.encode_data(xt, maxlen, vocab)
x_test = data_helpers.encode_data(x_test, maxlen, vocab)

print('Chars vocab: {}'.format(alphabet))
print('Chars vocab size: {}'.format(vocab_size))
print('X_train.shape: {}'.format(xt.shape))
model.summary()
print('Fit model...')
model.fit(xt, yt,
          validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, shuffle=True)

if save:
    print('Saving model params...')
    json_string = model.to_json()
    with open(model_name_path, 'w') as f:
        json.dump(json_string, f)

    model.save_weights(model_weights_path)
