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

import deepboof.DeepUnitTest;
import deepboof.Function;
import deepboof.factory.FactoryForwards;
import deepboof.tensors.Tensor_F32;

import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForwardSpatialConvolve2D_F32 extends ChecksForwardSpatialWindow_F32<ConfigConvolve2D> {

	public ChecksForwardSpatialConvolve2D_F32() {
		numberOfConfigurations = 3;
	}

	@Override
	public Function<Tensor_F32> createForwards(int which) {
		ConfigConvolve2D config = new ConfigConvolve2D();
		this.config = config;
		this.configPadding = new ConfigPadding();

		config.WW = 3;
		config.HH = 4;
		config.F = 6;

		switch( which ) {
			case 0:break;
			case 1:{
				config.WW = 1;
				config.HH = 1;
			}break;
			case 2:{
				configPadding.x0 = 1;
				configPadding.x1 = 2;
				configPadding.y0 = 3;
				configPadding.y1 = 4;
			}break;

			default:
				throw new RuntimeException("Unexpected");
		}

		return createForwards(config, configPadding);
	}

	protected abstract Function<Tensor_F32> createForwards( ConfigConvolve2D configConv,
															ConfigPadding configPadding );

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(2, parameters.size());

		int[] weights = parameters.get(0);
		int[] bias = parameters.get(1);

		int C = input[0];

		DeepUnitTest.assertEquals(WI(config.F,C,config.HH,config.WW),weights);
		DeepUnitTest.assertEquals(WI(config.F),bias);
	}

	@Override
	public SpatialPadding2D_F32 createPadding(int which) {
		return (SpatialPadding2D_F32)FactoryForwards.spatialPadding(configPadding,Tensor_F32.class);
	}

	@Override
	public int inputToOutputChannelCount(int numInput) {
		return config.F;
	}

	@Override
	protected float[] computeExpected(Tensor_F32 input, List<Tensor_F32> parameters, int batch, int y, int x) {

		int C = input.length(1);

		Tensor_F32 weights = parameters.get(0);
		Tensor_F32 bias = parameters.get(1);

		float output[] = new float[config.F];

		for (int f = 0; f < config.F; f++) {
			float sum = 0;
			for (int c = 0; c < C; c++) {
				for (int i = 0; i < config.HH; i++) {
					for (int j = 0; j < config.WW; j++) {
						sum += input.get(batch, c,y+i,x+j)*weights.get(f,c,i,j);
					}
				}
			}
			sum += bias.get(f);

			output[f] = sum;
		}

		return output;
	}

}
