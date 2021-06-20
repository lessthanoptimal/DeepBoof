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
import deepboof.DeepUnitTest;
import deepboof.impl.forward.standard.FunctionLinear_F32;
import deepboof.misc.TensorFactory_F32;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F32;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static deepboof.misc.TensorOps.WT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Peter Abeles
 */
public abstract class ChecksFunctionLinear_F32 extends ChecksForward<Tensor_F32> {

	protected int numOutputs = 7;

	@Test
	public void forward() {
		for( boolean sub : new boolean[]{false,true}) {
			int mini = 4;
			int A = 4, B = 6;
			int shape[] = new int[]{mini, A, B};

			FunctionLinear_F32 alg = new FunctionLinear_F32(numOutputs);

			alg.initialize(new int[]{A,B});

			Tensor_F32 input = TensorFactory_F32.random(random,sub,shape);
			Tensor_F32 output = TensorFactory_F32.random(random,sub,mini,numOutputs);
			Tensor_F32 weights = TensorFactory_F32.random(random,sub,alg.getParameterShapes().get(0));
			Tensor_F32 bias = TensorFactory_F32.random(random,sub,alg.getParameterShapes().get(1));

			alg.setParameters(WT(weights,bias));
			alg.forward(input, output);

			Tensor_F32 expected = TensorFactory_F32.zeros(sub?random:null,mini,numOutputs);

			int D = A*B;

			for (int batch = 0; batch < mini; batch++) {
				for (int o = 0; o < numOutputs; o++) {
					float total = 0;

					int indexIn = input.idx(batch,0,0);

					int indexW = weights.idx(o,0);

					for (int i = 0; i < D; i++) {
						total += input.d[indexIn++] * weights.d[indexW++];
					}
					expected.d[expected.idx(batch,o)] = total + bias.d[bias.idx(o)];
				}
			}

			DeepUnitTest.assertEquals(expected,output, DeepBoofConstants.TEST_TOL_F32);
		}
	}

	@Override
	public List<Case> createTestInputs() {

		List<Case> inputs = new ArrayList<>();

		inputs.add( new Case(1) );
		inputs.add( new Case(1,1) );
		inputs.add( new Case(3,6,2) );
		inputs.add( new Case(2,1,8) );
		inputs.add( new Case(2,3,4,2) );

		return inputs;
	}

	@Override
	protected void checkTensorType(Class<Tensor_F32> type) {
		assertTrue(Tensor_F32.class == type);
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		// total number of inputs
		int N = TensorOps.tensorLength(input);

		assertEquals(2,parameters.size());

		DeepUnitTest.assertEquals(WI(numOutputs,N), parameters.get(0));
		DeepUnitTest.assertEquals(WI(numOutputs), parameters.get(1));
	}

	@Override
	protected void checkOutputShapes(int[] input, int[] output) {
		DeepUnitTest.assertEquals(WI(numOutputs), output);
	}
}