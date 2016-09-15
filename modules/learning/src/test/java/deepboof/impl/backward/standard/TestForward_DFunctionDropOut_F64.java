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

import deepboof.DeepUnitTest;
import deepboof.misc.TensorFactory;
import deepboof.misc.TensorOps_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.Random;

/**
 * @author Peter Abeles
 */
public class TestForward_DFunctionDropOut_F64 {

	Random rand = new Random(234);

	TensorFactory<Tensor_F64> factory = new TensorFactory<>(Tensor_F64.class);


	/**
	 * Tests to see if it converges towards the expected average
	 */
	@Test
	public void learning() {

		double drop = 0.3;

		DFunctionDropOut_F64 alg = new DFunctionDropOut_F64(1234,drop);

		Tensor_F64 input = factory.random(rand,false,5.0,6.0,3,4);
		Tensor_F64 output = input.createLike();
		Tensor_F64 average = input.createLike();

		alg.initialize(4);
		alg.learning();
		int N = 1000;
		for (int i = 0; i < N; i++) {
			alg.forward(input,output);

			TensorOps_F64.elementAdd(output,average,average);
		}
		TensorOps_F64.elementMult(average,1.0/N);
		TensorOps_F64.elementMult(input,1.0-drop);

		DeepUnitTest.assertEquals(input,average, 0.2);
	}

	@Test
	public void evaluating() {
		double drop = 0.3;

		DFunctionDropOut_F64 alg = new DFunctionDropOut_F64(1234,drop);

		Tensor_F64 input = factory.random(rand,false,5.0,6.0,3,4);
		Tensor_F64 output = input.createLike();

		alg.initialize(4);
		alg.evaluating();
		alg.forward(input,output);

		TensorOps_F64.elementMult(input,1.0-drop);
		DeepUnitTest.assertEquals(input,output, 0.2);
	}
}
