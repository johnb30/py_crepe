py_crepe
========

This is an re-implementation of the
[Crepe](https://github.com/zhangxiangxiao/Crepe) character-level convolutional
neural net model described in this [paper](https://arxiv.org/abs/1509.01626).
The re-implementation is done using the Python library Keras(v2.1.5), using the Tensorflow-gpu(v1.6)
backend. Keras sits on top of Tensorflow so can make use of cuDNN for faster models.

Details
-------

The model implemented here follows the one described in the paper rather
closely. This means that the vocabulary is the same, fixed vocabulary described
in the paper, as well as using stochastic gradient descent as the optimizer.
Running the model for 9 epochs returns results in line with those described in the paper. 

It's relatively easy to modify the code to use a different vocabulary, e.g.,
one learned from the training data, and to use different optimizers such as
`adam`. 

Dataset
-------
https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv

Running
-------

The usual command to run the model is (for tensorflow-gpu backend):

```
python main.py
```

Score
-------
With current code it reaches similar level of accuracy stated on the paper.
We get test accuracy above 0.88 after 10 epoch of running (with Adam(lr=0.001) optimizer)

Note
-------
We had to specify the `kernel_initializer` for all the `Convolution1D` layers as
`RandomNormal(mean=0.0, stddev=0.05, seed=None)` as with the default initializer for Keras(v2.1.5) the model cannot
converge. I don't exactly know the reason behind this but the lesson learned is that: initialization matters!