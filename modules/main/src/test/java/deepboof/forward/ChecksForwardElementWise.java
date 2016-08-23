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

import deepboof.Function;
import deepboof.Tensor;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForwardElementWise<T extends Tensor<T>>
	extends ChecksForward<T>
{

	/**
	 * used to adjust the scale of randomly generated input tensors
	 */
	protected double inputScale = 1.0;

	/**
	 * Creates random tensors which are then used to test the {@link Function#forward} function.
	 */
	@Test
	public void checkForwardRandomInput() {

		for (int algConfig = 0; algConfig < numberOfConfigurations ; algConfig++) {
			Function<T> alg = createForwards(algConfig);

			List<Case> testCases = createTestInputs();

			for (boolean sub : new boolean[]{false, true}) {
				for (Case testCase : testCases) {
					T inputTensor = tensorFactory.randomM(random, sub, testCase.minibatch, testCase.inputShape);

					alg.initialize(testCase.inputShape);

					List<T> parameters = createParameters(alg, inputTensor);
					T outputTensor = tensorFactory.randomM(random, sub, testCase.minibatch, alg.getOutputShape());

					alg.setParameters(parameters);
					alg.forward(inputTensor, outputTensor);
					checkForwardResults(inputTensor, outputTensor);
				}
			}
		}
	}

	public abstract List<T> createParameters(Function<T> function , T input );

	public abstract void checkForwardResults(T input , T output );

	@Override
	public List<Case> createTestInputs() {
		List<Case> valid = new ArrayList<>();

		valid.add( new Case(WI()));
		valid.add( new Case(WI(1)));
		valid.add( new Case(WI(4,1,2)));
		valid.add( new Case(WI(2,4,5,2)));

		return valid;
	}

	@Override
	protected void checkOutputShapes(int[] input, int[] output) {

		assertEquals(input.length, output.length);

		for (int i = 0; i < output.length; i++) {
			assertEquals(input[i],output[i]);
		}
	}
}
