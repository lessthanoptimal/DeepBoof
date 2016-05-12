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

package deepboof.forward;

import deepboof.DeepBoofConstants;
import deepboof.Function;
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.VTensor_F64;
import org.junit.Test;

import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 * Common checks for classes which sample spatial tensors using a moving window
 *
 * @author Peter Abeles
 */
public abstract class ChecksForwardSpatialWindow_F64<C extends ConfigSpatial>
		extends ChecksForwardSpatial_F64
{

	protected C config;
	protected ConfigPadding configPadding = new ConfigPadding();

	/**
	 * Will always be called after createBasic
	 */
	public abstract SpatialPadding2D_F64 createPadding(int which );

	/**
	 * Given the number of input channels, comput the number of output channels
	 */
	public abstract int inputToOutputChannelCount( int numInput );

	@Override
	public boolean areExceptionsExpected(Function<Tensor_F64> function , int[] input ) {

		int Ho = 1+ (input[1]-config.HH+configPadding.y0 +configPadding.y1) / config.periodY;
		int Wo = 1+ (input[2]-config.WW+configPadding.x0 +configPadding.x1) / config.periodX;

		return Ho <= 0 || Wo <= 0;
	}

	@Override
	protected void checkOutputShapes(int[] input, int[] output) {

		assertEquals(3,output.length);

		int Hp = input[1] + configPadding.y0 + configPadding.y1;
		int Wp = input[2] + configPadding.x0 + configPadding.x1;

		// direct computation of output width and height
		int expectedHeight = 0;
		for (int y = 0; y < Hp; y += config.periodY) {
			if( Hp - y >= config.HH )
				expectedHeight++;
		}

		int expectedWidth = 0;
		for (int x = 0; x < Wp; x += config.periodX) {
			if( Wp - x >= config.WW )
				expectedWidth++;
		}

		assertEquals(inputToOutputChannelCount(output[0]),output[0]);
		assertEquals(expectedHeight,output[1]);
		assertEquals(expectedWidth,output[2]);
	}

	/**
	 * Pass in random data and see if it produces the same output as brute for computation
	 */
	@Test
	public void checkOutputValues() {

		int N = 3;

		for( boolean sub : new boolean[]{false,true}) {
			List<int[]> inputShapes = createTestInputs();

			for (int config = 0; config < numberOfConfigurations; config++) {
				Function<Tensor_F64> alg = createForwards(config);
				SpatialPadding2D_F64 padding = createPadding(config);

				for( int[] inputShape : inputShapes ) {
					try {
						alg.initialize(inputShape);
					} catch( RuntimeException ignore ) {
						continue;
					}

					int outputShape[] = alg.getOutputShape();

					Tensor_F64 input = TensorFactory_F64.randomMM(random,sub,-1,1,WI(N,inputShape));
					Tensor_F64 output = TensorFactory_F64.randomMM(random,sub,-1,1,WI(N,outputShape));

					List<Tensor_F64> parameters = TensorFactory_F64.randomMM(random,sub,-1,-1,alg.getParameterShapes());

					alg.setParameters(parameters);
					alg.forward(input, output);

					padding.setInput(input);
					checkOutputValues(padding,input,parameters,output);
				}
			}
		}
	}

	private void checkOutputValues(SpatialPadding2D_F64 padding, Tensor_F64 input,
								   List<Tensor_F64> parameters, Tensor_F64 output) {
		int N = input.length(0);
		int C = input.length(1);
		int Hp = input.length(2) + configPadding.y0 + configPadding.y1;
		int Wp = input.length(3) + configPadding.x0 + configPadding.x1;

		int numberOfOutputChannels = inputToOutputChannelCount(C);

		padding.setInput(input);
		Tensor_F64 padded = virtualToDense(padding);

		for (int batch = 0; batch < N; batch++) {

				int outY = 0;
				for (int y = 0; y <= Hp - config.HH; y += config.periodY, outY++) {
					int outX = 0;
					for (int x = 0; x <= Wp - config.WW; x += config.periodX, outX++) {
						double expected[] = computeExpected(padded, parameters, batch, y, x);

						assertEquals(numberOfOutputChannels,expected.length);
						for (int channel = 0; channel < numberOfOutputChannels; channel++) {
							double found = output.get(batch, channel, outY, outX);
							assertEquals(expected[channel], found, DeepBoofConstants.TEST_TOL_F64);
						}
					}
					assertEquals(outX, output.length(3));
				}
				assertEquals(outY, output.length(2));
		}
	}

	private static Tensor_F64 virtualToDense( VTensor_F64 tensor ) {
		int N = tensor.length(0);
		int C = tensor.length(1);
		int rows = tensor.length(2);
		int cols = tensor.length(3);

		Tensor_F64 output = new Tensor_F64(N,C,rows,cols);

		for (int batch = 0; batch < N; batch++) {
			for (int channel = 0; channel < C; channel++) {
				for (int y = 0; y < rows; y++) {
					for (int x = 0; x < cols; x++) {
						output.d[output.idx(batch,channel,y,x)] = tensor.get(batch,channel,y,x);
					}
				}
			}
		}

		return output;
	}

	protected abstract double[] computeExpected(Tensor_F64 input, List<Tensor_F64> parameters,
												int batch, int y, int x);
}
