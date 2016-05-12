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

package deepboof.io.torch7;

import deepboof.Accuracy;
import deepboof.DeepUnitTest;
import deepboof.Function;
import deepboof.Tensor;
import deepboof.graph.ForwardSequence;
import deepboof.impl.forward.standard.*;
import deepboof.io.torch7.struct.TorchObject;
import deepboof.misc.TensorOps;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;
import static deepboof.misc.TensorOps.TH;
import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.*;

/**
 * @author Peter Abeles
 */
public class TestConvertTorchToBoofForward {

	File pathToData = new File(TensorOps.pathToRoot(),"modules/io/src/test/torch7/data");

	@Test
	public void relu() {
		checkFunction("relu/F64",ActivationReLU_F64.class);
		checkFunction("relu/F32",ActivationReLU_F32.class);
	}

	@Test
	public void sigmoid() {
		checkFunction("sigmoid/F64",ActivationSigmoid_F64.class);
		checkFunction("sigmoid/F32",ActivationSigmoid_F32.class);
	}

	@Test
	public void tanh() {
		checkFunction("tanh/F64",ActivationTanH_F64.class);
		checkFunction("tanh/F32",ActivationTanH_F32.class);
	}

	@Test
	public void linear() {
		checkFunction("linear/F64",FunctionLinear_F64.class);
		checkFunction("linear/F32",FunctionLinear_F32.class);
	}

	@Test
	public void cudaLinear() {
		checkFunction("cuda_linear/cuda",FunctionLinear_F32.class);
	}

	@Test
	public void batchNormalization() {
		checkFunction("batch_normalization/F64",FunctionBatchNorm_F64.class);
		checkFunction("batch_normalization/F32",FunctionBatchNorm_F32.class);
	}

	@Test
	public void spatialConvolution() {
		checkFunction("spatial_convolution/F64", SpatialConvolve2D_F64.class);
		checkFunction("spatial_convolution/F32", SpatialConvolve2D_F32.class);
	}

	@Test
	public void spatialMaxPooling() {
		checkFunction("spatial_max_pooling/F64", SpatialMaxPooling_F64.class);
		checkFunction("spatial_max_pooling/F32", SpatialMaxPooling_F32.class);
	}

	@Test
	public void spatialBatchNorm() {
		checkFunction("spatial_batch_normalization/F64", SpatialBatchNorm_F64.class);
		checkFunction("spatial_batch_normalization/F32", SpatialBatchNorm_F32.class);
	}

	@Test
	public void sequential() {
		checkSequence("sequential/F64");
		checkSequence("sequential/F32");
	}

	@Test
	public void dropout() {
		checkFunction("dropout/F64", null);
		checkFunction("dropout/F32", null);
	}

	@Test
	public void spatialDropout() {
		checkFunction("spatial_dropout/F64", null);
		checkFunction("spatial_dropout/F32", null);
	}

	private void checkFunction(String directory , Class functionClass ) {
		File pathToOp = new File(pathToData,directory);
		if( !pathToOp.exists() ) {
			System.err.println("Can't find torch generated test data.  To generate data run the following Linux shell script");
			System.err.println();
			System.err.println("DeepBoof/modules/io/src/test/torch7/generate_all.sh");
			System.err.println();
			fail("Missing torch data for "+directory);
		}

		int count = 0;
		for( File d : pathToOp.listFiles() ) {
			if( !d.isDirectory() )
				continue;

			Tensor input = convert(read(new File(d,"input")));
			FunctionAndParameters fap = convert(read(new File(d,"operation")));
			Tensor expected = convert(read(new File(d,"output")));

			if( fap != null ) {
				Function function = fap.getFunction();
				if (functionClass != null)
					assertTrue("Unexpected class type. " + function.getClass().getSimpleName(),
							function.getClass() == functionClass);

				int N = input.length(0);

				function.initialize(TH(input.getShape()));
				Tensor found = input.create(WI(N, function.getOutputShape()));
				function.setParameters(fap.parameters);
				function.forward(input, found);
				DeepUnitTest.assertEquals(expected,found, Accuracy.STANDARD);
			} else {
				// there is no function, so the function is supposed to be pointless and input should be
				// the same as output
				DeepUnitTest.assertEquals(expected,input, Accuracy.STANDARD);
			}

			count++;
		}
		assertTrue(count>0);
	}

	private void checkSequence(String directory ) {
		File pathToOp = new File(pathToData,directory);

		int count = 0;
		for( File d : pathToOp.listFiles() ) {
			if( !d.isDirectory() )
				continue;

			Tensor input = convert(read(new File(d,"input")));
			SequenceAndParameters sap = convert(read(new File(d,"operation")));
			Tensor expected = convert(read(new File(d,"output")));

			ForwardSequence forward = new ForwardSequence(sap.sequence,sap.type);

			int N = input.length(0);

			forward.initialize(TH(input.getShape()));
			forward.setParameters(sap.parameters);
			Tensor found = input.create(WI(N,forward.getOutputShape()));
			forward.process(input,found);

			DeepUnitTest.assertEquals(expected,found, Accuracy.STANDARD);
			count++;
		}
		assertTrue(count>0);
	}

	private <T extends TorchObject>T read( File path ) {
		try {
			List<TorchObject> found = new ParseBinaryTorch7().parse(path);
			assertEquals(1,found.size());
			return (T)found.get(0);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
