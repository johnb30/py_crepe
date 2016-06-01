py_crepe
========

This is an re-implementation of the
[Crepe](https://github.com/zhangxiangxiao/Crepe) character-level convolutional
neural net model described in this [paper](https://arxiv.org/abs/1509.01626).
The re-implementation is done using the Python library Keras, using the Theano
backend. Keras sits on top of Theano so can make use of cuDNN and CNMeM for faster
models. 

Details
-------

The model implemented here follows the one described in the paper rather
closely. This means that the vocabulary is the same, fixed vocabulary described
in the paper, as well as using stochastic gradient descent as the optimizer.
Running the model for 9 epochs returns results in line with those described in the paper. 

It's relatively easy to modify the code to use a different vocabulary, e.g.,
one learned from the training data, and to use different optimizers such as
`adam`. 

Running
-------

The usual command to run the model is:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1.0 python main.py
```

You can also set these parameters in a `~/.theanorc` file so you don't have to
run them each time. The most important parameters, other than running it on a
GPU, is the CNMeM allocation variable, which allows for much faster runtimes.
When combined with cuDNN there's a rather drastic speedup when compared to the
original Lua/Torch implementation.
