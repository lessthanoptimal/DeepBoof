== Introduction

The example below shows you how to train a network in Torch and create logs which can be read in by
visualization tools in DeepBoof.  These visualization tools are intended to help you optimize your
network better and understand its behavior better.

== Requirements

Install the following first:

luarocks install luaposix

== Grid Search

To generate a randomized search of parameter space for best optimization parameters invoke:

th run_grid.lua -g spread_adam --search adam

Since the default network is very simple it should run fast even on a computer without Cuda installed.
If you have cuda installed you can invoke the line below instead and it will run even faster.

th run_grid.lua -g spread_adam --search adam --type cuda

Then let this run for an hour or two.  See in code comments for the details about what it's doing.

== Single Run

BLAH


== TODO LIST

* Use same data set cifar data as java examples
* Add preprocessing steps from VGG example?
*