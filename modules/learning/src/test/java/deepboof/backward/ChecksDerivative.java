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
import deepboof.tensors.Tensor_F64;
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

	public boolean verbose = false;

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
			if( verbose )
				System.out.println("ALG Config "+algConfig);
			DFunction<T> alg = createBackwards(algConfig);
			alg.learning(); // algorithms which obey this flag require it to be in learning mode

			numeric.setFunction(alg);

			List<Case> testCases = createTestInputs();

			for (boolean sub : new boolean[]{false, true}) {
				for (Case testCase : testCases) {
					if( verbose )
						System.out.println("sub "+sub+"  input.length "+testCase.inputShape.length);
					T inputTensor = tensorFactory.randomM(random, sub, testCase.minibatch, testCase.inputShape);

					alg.initialize(testCase.inputShape);

					List<T> parameters = createParameters(alg, inputTensor);

					T outputTensor = tensorFactory.randomM(random, sub, testCase.minibatch, alg.getOutputShape());
					T dout = tensorFactory.randomM(random, sub, testCase.minibatch, alg.getOutputShape());

					// User the numerical gradient as ground truth for the gradient
					T expectedXD = tensorFactory.randomM(random, sub, testCase.minibatch, testCase.inputShape);
					List<T> expectedWD = createParameters(alg, inputTensor);
//					System.out.println("===== NUMERICAL GRADIENT START");
					numeric.differentiate(inputTensor,parameters,dout,expectedXD,expectedWD);
//					System.out.println("===== NUMERICAL GRADIENT STOP");

					// invoke the forwards pass first.  Some algorithms require it be called first
					alg.setParameters(parameters);
					// to compute the gradient
					alg.forward(inputTensor, outputTensor);

					// compute the gradient using the function being tested
					T foundXD = tensorFactory.randomM(random, sub, testCase.minibatch, testCase.inputShape);
					List<T> foundWD = createParameters(alg, inputTensor);
					alg.backwards(inputTensor,dout,foundXD,foundWD);

					if( verbose ) {
						System.out.println("Comparision of expected and found XD");
						for (int i = 0; i < expectedXD.length(); i++) {
							double e = ((Tensor_F64) expectedXD).d[i];
							double f = ((Tensor_F64) foundXD).d[i];
							System.out.printf("%6.2e   %6.2e\n", e, f);
						}

						System.out.print("     Input Shape  [ ");
						for (int i = 0; i < inputTensor.shape.length; i++) {
							System.out.print(" " + inputTensor.shape[i]);
						}
						System.out.println(" ]");
					}

					// compare results
					DeepUnitTest.assertEquals(expectedXD,foundXD, tolerance );
					for (int i = 0; i < expectedWD.size(); i++) {
						T e = expectedWD.get(i);
						T f = foundWD.get(i);
						DeepUnitTest.assertEquals(e, f, tolerance );
					}
				}
			}
		}
	}

	public abstract List<T> createParameters(DFunction<T> function , T input );
}
