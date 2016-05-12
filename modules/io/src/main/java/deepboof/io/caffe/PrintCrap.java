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
import com.google.protobuf.TextFormat;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class PrintCrap {

	public static CharSequence fromFile(String filename) throws IOException {
		FileInputStream fis = new FileInputStream(filename);
		FileChannel fc = fis.getChannel();

		// Create a read-only CharBuffer on the file
		ByteBuffer bbuf = fc.map(FileChannel.MapMode.READ_ONLY, 0,
				(int) fc.size());
		return Charset.forName("8859_1").newDecoder().decode(bbuf);
	}

	public static void main(String[] args) throws IOException {

		String path = "data/caffe_models/alexnet/deploy.prototxt";

		Caffe.NetParameter.Builder builder = Caffe.NetParameter.newBuilder();

		TextFormat.getParser().merge(fromFile(path),builder);

		System.out.println("name = "+builder.getName());

		List<Caffe.LayerParameter> layers = builder.getLayerList();

		System.out.println("Total layers = "+layers.size());

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
					System.out.println("  size    = "+param.getKernelSize(j));
				}
				System.out.println("  num out = "+param.getNumOutput());
				for (int j = 0; j < param.getStrideCount(); j++) {
					System.out.println("  stride  = "+param.getStride(j));
				}
			}
		}

//		System.out.println(builder);
	}
}
