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

package deepboof.backward;

import deepboof.Accuracy;
import deepboof.DFunction;
import deepboof.DeepUnitTest;
import deepboof.Tensor;
import deepboof.factory.FactoryBackwards;
import deepboof.forward.ChecksGenericFunction;
import deepboof.misc.TensorFactory;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

/**
 * @author Peter Abeles
 */
public abstract class ChecksDerivative<T extends Tensor<T>>
	extends ChecksGenericFunction<T>
{

	protected FactoryBackwards<T> factoryD;

	protected Accuracy tolerance = Accuracy.RELAXED_A;

	public abstract DFunction<T> createBackwards(int type );

	@Before
	public void before() {
		tensorFactory = new TensorFactory<>(createBackwards(0).getTensorType());
		factoryD = new FactoryBackwards<>(createBackwards(0).getTensorType());
	}

		/**
	 * Tests the {@link DFunction#backwards}
	 */
	@Test
	public void checkBackwardsRandomInput() {

		NumericalGradient<T> numeric = factoryD.createNumericalGradient();

		for (int algConfig = 0; algConfig < numberOfConfigurations ; algConfig++) {
			DFunction<T> alg = createBackwards(algConfig);

			numeric.setFunction(alg);

			List<int[]> inputShapes = createTestInputs();

			for (boolean sub : new boolean[]{false, true}) {
				for (int[] input : inputShapes) {
					T inputTensor = tensorFactory.randomM(random, sub, minibatch, input);

					alg.initialize(input);

					List<T> parameters = createParameters(alg, inputTensor);

					T outputTensor = tensorFactory.randomM(random, sub, minibatch, alg.getOutputShape());
					T dout = tensorFactory.randomM(random, sub, minibatch, alg.getOutputShape());

					// User the numerical gradient as ground truth for the gradient
					T expectedXD = tensorFactory.randomM(random, sub, minibatch, input);
					List<T> expectedWD = createParameters(alg, inputTensor);
					numeric.differentiate(inputTensor,parameters,dout,expectedXD,expectedWD);

					// invoke the forwards pass first.  Some algorithms require it be called first
					alg.setParameters(parameters);
					alg.forward(inputTensor, outputTensor);
					// compute the gradient using the function being tested
					T foundXD = tensorFactory.randomM(random, sub, minibatch, input);
					List<T> foundWD = createParameters(alg, inputTensor);
					alg.backwards(inputTensor,dout,foundXD,foundWD);

					// compare results
					DeepUnitTest.assertEquals(expectedXD,foundXD, tolerance );
				}
			}
		}
	}

	public abstract List<T> createParameters(DFunction<T> function , T input );
}
