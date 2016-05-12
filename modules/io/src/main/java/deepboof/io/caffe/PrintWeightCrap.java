/*
 * Copyright (c) 2016, Peter Abeles. All Rights Reserved.
 *
 * This file is part of DeepBoof
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package deepboof.io.caffe;

import caffe.Caffe;
import com.google.protobuf.CodedInputStream;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class PrintWeightCrap {

	public static CharSequence fromFile(String filename) throws IOException {
		FileInputStream fis = new FileInputStream(filename);
		FileChannel fc = fis.getChannel();

		// Create a read-only CharBuffer on the file
		ByteBuffer bbuf = fc.map(FileChannel.MapMode.READ_ONLY, 0,
				(int) fc.size());
		return Charset.forName("8859_1").newDecoder().decode(bbuf);
	}

	static void tabString( String message ) {
		String lines[] = message.split("\n");

		for( String line : lines ) {
			System.out.println("  "+line);
		}
	}

	static void printLayers( List<caffe.Caffe.V1LayerParameter> layers ) {

		System.out.println("---------- Total V1LayerParameter = "+layers.size());
		for (int i = 0; i < layers.size(); i++) {
			Caffe.V1LayerParameter layer = layers.get(i);
			System.out.println("------------------------------------------------------");
			System.out.println("name         = " + layer.getName());
			System.out.println("type         = " + layer.getType());
			System.out.println("has data     = " + layer.hasDataParam());

			List<String> bottoms = layer.getBottomList();
			List<String> tops = layer.getTopList();

			for( String s : tops ) {
				System.out.println("top          = "+s);
			}

			for( String s : bottoms ) {
				System.out.println("bottom       = "+s);
			}

			if( layer.getParamList().size() > 0 ) {
				List<String> paramList = layer.getParamList();
				System.out.println("parameter list  " + paramList.size());
				for (String w : paramList) {
					System.out.println("     " + w);
				}
			}
			if( layer.hasDataParam() ) {
				System.out.println("Data Param");
				caffe.Caffe.DataParameter param = layer.getDataParam();
				tabString(param.toString());
			}

			if( layer.hasConvolutionParam() ) {
				System.out.println("Convolution Param");
				caffe.Caffe.ConvolutionParameter param = layer.getConvolutionParam();
				tabString(param.toString());
			}

			if( layer.hasDropoutParam() ) {
				System.out.println("Dropout Param");
				caffe.Caffe.DropoutParameter param = layer.getDropoutParam();
				tabString(param.toString());
			}
			if( layer.hasInnerProductParam() ) {
				System.out.println("Inner Product Param");
				caffe.Caffe.InnerProductParameter param = layer.getInnerProductParam();
				tabString(param.toString());
			}

			if( layer.hasLrnParam() ) {
				System.out.println("LRN Param");
				caffe.Caffe.LRNParameter param = layer.getLrnParam();
				tabString(param.toString());
			}

			if( layer.hasPoolingParam() ) {
				System.out.println("Pooling Param");
				caffe.Caffe.PoolingParameter param = layer.getPoolingParam();
				tabString(param.toString());
			}

			List<caffe.Caffe.BlobProto> blobs = layer.getBlobsList();

			if( blobs.size() > 0 ) {
				System.out.println("Blobs  size = "+blobs.size());
				for( caffe.Caffe.BlobProto blob : blobs ) {
					System.out.println("   --- blob");
					if( blob.hasShape()) {
						caffe.Caffe.BlobShape shape = blob.getShape();
						System.out.print("  shape = ");
						for (int j = 0; j < shape.getDimCount(); j++) {
							System.out.print(" "+shape.getDim(j));
						}
						System.out.println();
					}
					if( blob.hasNum() )
						System.out.println("  num = "+blob.getNum());
					if( blob.hasChannels() )
						System.out.println("  channels = "+blob.getChannels());
					if( blob.hasHeight() )
						System.out.println("  height = "+blob.getHeight());
					if( blob.hasWidth() )
						System.out.println("  width = "+blob.getWidth());

					System.out.println("  data count = "+blob.getDataCount());
					System.out.println("  diff count = "+blob.getDiffCount());
				}
			}

		}
		System.out.println();
	}

	public static void printLayer( List<caffe.Caffe.LayerParameter> layers ) {
		System.out.println("---------- Total LayerParameter = "+layers.size());
		for (int i = 0; i < layers.size(); i++) {
			Caffe.LayerParameter layer = layers.get(i);
			System.out.println("------------------------------------------------------");
			System.out.println("name         = "+layer.getName());
			System.out.println("type         = "+layer.getType());

			List<String> bottoms = layer.getBottomList();
			List<String> tops = layer.getTopList();

			for( String s : tops ) {
				System.out.println("top          = "+s);
			}

			for( String s : bottoms ) {
				System.out.println("bottom       = "+s);
			}

			// optimization parameters
//			List<Caffe.ParamSpec> params = layer.getParamList();
//			System.out.println("total params = "+params.size());

			if( layer.hasConvolutionParam() ) {
				System.out.println("Convolution Parameters:");
				Caffe.ConvolutionParameter param = layer.getConvolutionParam();
				for (int j = 0; j < param.getKernelSizeCount(); j++) {
					System.out.println("  kernel size    = "+param.getKernelSize(j));
				}
				System.out.println("  num out = "+param.getNumOutput());
				for (int j = 0; j < param.getStrideCount(); j++) {
					System.out.println("  stride  = "+param.getStride(j));
				}
			}
		}
		System.out.println();
	}

	public static void main(String[] args) throws IOException {

		String path = "data/caffe_models/alexnet/bvlc_alexnet.caffemodel";
//		String path = "data/caffe_models/alexnet/imagenet_mean.binaryproto";

		CodedInputStream input = CodedInputStream.newInstance(new FileInputStream(path));
		input.setSizeLimit(512*1024*1024);

		System.out.println("Before");
		Caffe.NetParameter parameters = Caffe.NetParameter.parseFrom(input);
		System.out.println("After");

		System.out.println("name = "+parameters.getName());

		System.out.println("   getInputDimCount()  = "+parameters.getInputDimCount());
		System.out.println("   getLayerList().size() = "+parameters.getLayerList().size());
		System.out.println("   getLayersList().size() = "+parameters.getLayersList().size());
		System.out.println("   getInputShapeCount() = "+parameters.getInputShapeCount());

		List<Caffe.LayerParameter> layers = parameters.getLayerList();

		printLayers(parameters.getLayersList());
		printLayer(parameters.getLayerList());


	}
}
