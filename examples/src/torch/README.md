## Introduction

The example below shows you how to train a network in Torch and create logs which can be read in by
visualization tools in DeepBoof.  These visualization tools are intended to help you optimize your
network better and understand its behavior better.

# Requirements

First make sure you have Torch installed then add the luaposix package to your Torch setup.  This code 
has only been tested in Linux and run through the command-line.

* [Install Torch 7](http://torch.ch/docs/getting-started.html#_)
* luarocks install luaposix

Optional For CUDA
* luarocks install cutorch
* luarocks install cunn

## Data Normalization

DeepBoof is used to normalize the raw input data and this must be done before any training can
be performed. The first command below will generate a description of the statistical properties
of the images in YUV color space.  The second command will first convert the image into 
YUV color space then apply the normalization such that they have a mean of zero and standard
deviation of one.

```bash
gradle exampleRun -Pwhich=ExampleLearnNormalizationCifar10
gradle exampleRun -Pwhich=ExampleApplyNormalizeCifar10
```

Inside Deepboof/examples there will now be train_normalized_cifar10.t7 and train_normalized_cifar10.t7 files.

## Training Parameter Grid Search

Parameter search is used to determine that the best set of values to use when training.  This can be tricky
since there is no universally best value for parameters like the optimization step size.  What the code
below does is loads a range of values to search over from torch/grids and runs it on a subset of the input
data.  50 sample points is probably a good minimum number.  Select the one with the best training and use
its values "Single Run" example below.

To generate a randomized search of parameter space for best optimization parameters invoke:

```bash
th run_grid.lua -g spread_adam --search adam
```

Since the default network is very simple it should run fast even on a computer without Cuda installed.
If you have cuda installed you can invoke the line below instead and it will run even faster.

```bash
th run_grid.lua -g spread_adam --search adam --type cuda
```

Then let this run for an hour or two.  See in code comments for the details about what it's doing.

## Single Run

BLAH


## TODO LIST

* Use same data set cifar data as java examples
* Add preprocessing steps from VGG example?
*
