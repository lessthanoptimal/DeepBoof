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
import deepboof.forward.ChecksForward;
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
public class TestFunctionBatchNorm_F64 extends ChecksForward<Tensor_F64> {

	// variable  to keep track of which configuration was
	boolean requiresGamma;

	int N = 5; // number of mini-batch
	int d0 = 3; // arbitrary other dimensions
	int d1 = 2;
	int D = d0 * d1; // total length of other dimensions

	double EPS = (double)1e-4;

	public TestFunctionBatchNorm_F64() {
		super(2);
	}

	@Test
	public void check_two() {

		for( boolean sub : new boolean[]{false,true}) {
			FunctionBatchNorm_F64 alg = new FunctionBatchNorm_F64(false);

			alg.initialize(d0, d1);
			alg.setEPS(EPS);

			Tensor_F64 input = TensorFactory_F64.random(random, sub, N, d0, d1);
			Tensor_F64 params = TensorFactory_F64.random(random, sub, d0, d1, 2);
			Tensor_F64 output = TensorFactory_F64.random(random, sub, N, d0, d1);

			Tensor_F64 expected = new Tensor_F64(N, d0, d1);

			for (int batch = 0; batch < N; batch++) {
				int indexIn = input.idx(batch, 0, 0);
				int indexP = params.idx(0,0,0);
				int indexOut = expected.idx(batch,0,0);

				for (int i = 0; i < D; i++) {
					double m = params.d[indexP++];
					double v = params.d[indexP++];

					expected.d[indexOut++] = (input.d[indexIn++]-m) / Math.sqrt(v+EPS);
				}
			}

			alg.setParameters(WT(params));
			alg.forward(input, output);

			DeepUnitTest.assertEquals(expected,output, DeepBoofConstants.TEST_TOL_F64);
		}
	}

	@Test
	public void check_four() {
		for( boolean sub : new boolean[]{false,true}) {
			FunctionBatchNorm_F64 alg = new FunctionBatchNorm_F64(true);

			alg.initialize(d0, d1);
			alg.setEPS(EPS);

			Tensor_F64 input = TensorFactory_F64.random(random, sub, N, d0, d1);
			Tensor_F64 params = TensorFactory_F64.random(random, sub, d0, d1, 4);
			Tensor_F64 output = TensorFactory_F64.random(random, sub, N, d0, d1);

			Tensor_F64 expected = new Tensor_F64(N, d0, d1);

			for (int batch = 0; batch < N; batch++) {
				int indexIn = input.idx(batch, 0, 0);
				int indexP = params.idx(0,0,0);
				int indexOut = expected.idx(batch,0,0);

				for (int i = 0; i < D; i++) {
					double m = params.d[indexP++];
					double v = params.d[indexP++];
					double g = params.d[indexP++];
					double b = params.d[indexP++];

					expected.d[indexOut++] = ((input.d[indexIn++]-m) / Math.sqrt(v+EPS))*g+b;
				}
			}

			alg.setParameters(WT(params));
			alg.forward(input, output);

			DeepUnitTest.assertEquals(expected,output, DeepBoofConstants.TEST_TOL_F64);
		}
	}

	@Override
	public Function<Tensor_F64> createForwards(int which) {
		requiresGamma = which != 0;
		if( which == 0 )
			return new FunctionBatchNorm_F64(false);
		else
			return new FunctionBatchNorm_F64(true);
	}

	@Override
	public List<Case> createTestInputs() {
		List<Case> inputs = new ArrayList<>();

		inputs.add( new Case( 5,1 ));
		inputs.add( new Case( 5,2,3,4 ));

		return inputs;
	}

	@Override
	protected void checkTensorType(Class<Tensor_F64> type) {
		assertTrue(Tensor_F64.class == type);
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(1,parameters.size());

		int expected[] = new int[input.length+1];
		System.arraycopy(input,0,expected,0,input.length);
		expected[input.length] = requiresGamma ? 4 : 2;;

		DeepUnitTest.assertEquals(expected, parameters.get(0));
	}

	@Override
	protected void checkOutputShapes(int[] input, int[] output) {
		DeepUnitTest.assertEquals(output, input);
	}
}
