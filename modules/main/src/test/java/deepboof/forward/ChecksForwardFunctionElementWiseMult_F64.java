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
import deepboof.tensors.Tensor_F64;
import org.junit.Assert;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForwardFunctionElementWiseMult_F64 extends ChecksForwardElementWise_F64 {

	public ChecksForwardFunctionElementWiseMult_F64() {
		inputScale = 2.0;
	}

	protected double scalar = 0.25;

	@Override
	public List<Tensor_F64> createParameters(Function<Tensor_F64> function, Tensor_F64 input) {
		return new ArrayList<>();
	}

	@Override
	public void checkForwardResults(Tensor_F64 input, Tensor_F64 output) {
		int N = input.length();

		assertTrue(N>0); // sanity check input

		for (int i = 0; i < N; i++) {
			double value = input.getAtIndex(i);
			double expected = scalar*value;
			Assert.assertEquals(expected,output.getAtIndex(i), DeepBoofConstants.TEST_TOL_F64);
		}
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(0,parameters.size());
	}
}