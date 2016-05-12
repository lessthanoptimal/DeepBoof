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
import deepboof.tensors.Tensor_F32;
import org.junit.Assert;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForwardFunctionElementWiseMult_F32 extends ChecksForwardElementWise_F32 {

	public ChecksForwardFunctionElementWiseMult_F32() {
		inputScale = 2.0f;
	}

	protected float scalar = 0.25f;

	@Override
	public List<Tensor_F32> createParameters(Function<Tensor_F32> function, Tensor_F32 input) {
		return new ArrayList<>();
	}

	@Override
	public void checkForwardResults(Tensor_F32 input, Tensor_F32 output) {
		int N = input.length();

		assertTrue(N>0); // sanity check input

		for (int i = 0; i < N; i++) {
			float value = input.getAtIndex(i);
			float expected = scalar*value;
			Assert.assertEquals(expected,output.getAtIndex(i), DeepBoofConstants.TEST_TOL_F32);
		}
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(0,parameters.size());
	}
}