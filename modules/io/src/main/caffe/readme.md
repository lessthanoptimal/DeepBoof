## Copyrights

The following are not covered by the Deep Boof copyright and are owned by the Caffe project and covered by
the "Caffe.LICENSE" included here.  These files are are included for convenience and to help ensure compatibility
with Deep Boof.

* caffe.proto


## Code Generation

To generate the protobuff Java code for reading caffe models, invoke the following:

```bash
cd DeepBoof/modules/io/src/main/caffe
protoc -I=./ --java_out=../java/deepboof/io/ caffe.proto
```