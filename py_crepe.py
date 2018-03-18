from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal


def create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    # Define what the input shape looks like
    inputs = Input(shape=(maxlen,), dtype='int64')

    # Option one:
    # Uncomment following code to use a lambda layer to create a onehot encoding of a sequence of characters on the fly.
    # Holding one-hot encodings in memory is very inefficient.
    # The output_shape of embedded layer will be: batch x maxlen x vocab_size
    #
    import tensorflow as tf

    def one_hot(x):
        return tf.one_hot(x, vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)

    def one_hot_outshape(in_shape):
        return in_shape[0], in_shape[1], vocab_size

    embedded = Lambda(one_hot, output_shape=one_hot_outshape)(inputs)

    # Option two:
    # Or, simply use Embedding layer as following instead of use lambda to create one-hot layer
    # Think of it as a one-hot embedding and a linear layer mashed into a single layer.
    # See discussion here: https://github.com/keras-team/keras/issues/4838
    # Note this will introduce one extra layer of weights (of size vocab_size x vocab_size = 69*69 = 4761)
    # embedded = Embedding(input_dim=vocab_size, output_dim=vocab_size)(inputs)

    # All the convolutional layers...
    conv = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], kernel_initializer=initializer,
                         padding='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(embedded)
    conv = MaxPooling1D(pool_size=3)(conv)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=3)(conv1)

    conv2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv1)

    conv3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[3], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv2)

    conv4 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[4], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv3)

    conv5 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[5], kernel_initializer=initializer,
                          padding='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    # Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(inputs=inputs, outputs=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
