DeepBoof is a Java library for running deep neural networks on image data and other sources in Java.  
Additional tools include visualization and network training.  It has been designed to integrate with 
[Torch](http://torch.ch/) but does not depend it for its core functionality.

## Installation

[Gradle](http://gradle.org) is used to build DeepBoof and will automatically download all dependencies.

## Running

To get started classifying images simply load the Gradle project into you favorite IDE (probably IntelliJ
or Eclipse) and run ExampleClassifyVggCifar10.  This example downloads the network and testing data, then
starts the classifier.

Alternatively, you can just build and run any example from the command-line using Gradle directly:

```bash
gradle exampleRun -Pwhich=ExampleClassifyCifar10TestSet
```

## Future Work

DeepBoof is in an early state of development.  The following are areas it needs the most improvement in:

* Support for Caffe models (already started)
* Native implementations of functions
* GPU implementations of functions

