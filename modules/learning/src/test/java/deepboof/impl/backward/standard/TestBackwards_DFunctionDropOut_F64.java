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

package deepboof.impl.backward.standard;

import deepboof.DeepBoofConstants;
import deepboof.misc.TensorFactory;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DFunctionDropOut_F64 {
	Random rand = new Random(234);

	TensorFactory<Tensor_F64> factory = new TensorFactory<>(Tensor_F64.class);

	/**
	 * Check to see if there are zeros in the gradient at the expected location
	 */
	@Test
	public void checkZeros() {
		double drop = 0.3;

		DFunctionDropOut_F64 alg = new DFunctionDropOut_F64(1234,drop);

		Tensor_F64 input = factory.random(rand,false,5.0,6.0,3,4);
		Tensor_F64 output = input.createLike();

		alg.initialize(4);
		alg.learning();

		alg.forward(input,output);

		Tensor_F64 dout = factory.random(rand,false,5.0,6.0,3,4);
		Tensor_F64 gradientInput = input.createLike();

		alg.backwards(input,dout,gradientInput, new ArrayList<Tensor_F64>());

		for (int i = 0; i < 12; i++) {
			if( output.d[i] == 0 ) {
				assertEquals(0,gradientInput.d[i], DeepBoofConstants.TEST_TOL_F64);
			} else {
				assertEquals(dout.d[i],gradientInput.d[i], DeepBoofConstants.TEST_TOL_F64);
			}
		}

	}
}
