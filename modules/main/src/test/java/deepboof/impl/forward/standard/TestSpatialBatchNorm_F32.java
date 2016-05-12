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

package deepboof.impl.forward.standard;

import deepboof.DeepBoofConstants;
import deepboof.DeepUnitTest;
import deepboof.Function;
import deepboof.forward.ChecksForwardSpatial_F32;
import deepboof.tensors.Tensor_F32;
import org.junit.Test;

import java.util.List;

import static deepboof.misc.TensorFactory_F32.randomMM;
import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public class TestSpatialBatchNorm_F32 extends ChecksForwardSpatial_F32 {

	private boolean gammaBeta;

	float EPS = (float)1e-4;

	public TestSpatialBatchNorm_F32() {
		numberOfConfigurations = 2;
	}

	@Override
	public Function<Tensor_F32> createForwards(int which) {
		if( which == 0 ) {
			gammaBeta = false;
		} else {
			gammaBeta = true;
		}

		return new SpatialBatchNorm_F32(gammaBeta);
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(1,parameters.size());

		int N;
		if( gammaBeta ) {
			N = 4;
		} else {
			N = 2;
		}

		int C = input[0];

		DeepUnitTest.assertEquals(WI(C,N),parameters.get(0));
	}

	@Override
	protected void checkOutputShapes(int[] input, int[] output)
	{
		DeepUnitTest.assertEquals(input,output);
	}

	/**
	 * Checks values of output using a basic test
	 */
	@Test
	public void basic() {
		int N=2, C = 5, H = 3, W = 4;
		for (boolean gamma : new boolean[]{false, true}) {
			SpatialBatchNorm_F32 alg = new SpatialBatchNorm_F32(gamma);
			alg.setEPS(EPS);

			for( boolean sub : new boolean[]{false,true}) {
				Tensor_F32 input = randomMM(random, sub, -1, 1, WI(N, C, H, W));

				alg.initialize(WI(C,H,W));

				List<Tensor_F32> parameters = randomMM(random,sub,0.1f,2,alg.getParameterShapes());
				Tensor_F32 output = randomMM(random,sub,-1,1,WI(N,alg.getOutputShape()));

				alg.setParameters(parameters);
				alg.forward(input,output);

				compareToExpected(input,parameters.get(0),output);
			}
		}
	}

	private void compareToExpected(Tensor_F32 input, Tensor_F32 parameter, Tensor_F32 found) {
		int N = input.length(0);
		int C = input.length(1);
		int H = input.length(2);
		int W = input.length(3);

		int np = parameter.length(1);

		for (int batch = 0; batch < N; batch++) {
			for (int channel = 0; channel < C; channel++) {
				float mean = parameter.get(channel,0);
				float var = parameter.get(channel,1);

				float gamma = np == 4 ? parameter.get(channel,2) : 1;
				float beta = np == 4 ? parameter.get(channel,3) : 0;

				for (int i = 0; i < H; i++) {
					for (int j = 0; j < W; j++) {
						float f = found.get(batch,channel,i,j);
						float v = input.get(batch,channel,i,j);

						float expected = ((v-mean)/ (float)Math.sqrt(var+EPS) )*gamma + beta;

						assertEquals(expected,f, DeepBoofConstants.TEST_TOL_F32);
					}
				}
			}
		}
	}

}
