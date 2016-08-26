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
import deepboof.misc.TensorFactory;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * provides skeleton for performing very basic tests of the contract for Function
 *
 * @author Peter Abeles
 */
public abstract class ChecksForward<T extends Tensor<T>> extends ChecksGenericFunction<T> {

	public ChecksForward(int numberOfConfigurations) {
		super(numberOfConfigurations);
	}

	public ChecksForward() {
	}

	/**
	 * Create a Function for testing very basic functionality
	 */
	public abstract <F extends Function<T>> F createForwards(int which );

	/**
	 * If true is returned then an exception is expected on init() when this combination of function and input
	 * is passed in
	 */
	public boolean areExceptionsExpected(Function<T> function , int[] input ) {
		return false;
	}

	@Before
	public void before() {
		tensorFactory = new TensorFactory<>(createForwards(0).getTensorType());
	}

	/**
	 * Checks to see if the shape it is initialized with will work when processed in the forwards direction.
	 * Just checks to see if an exception is thrown
	 */
	@Test
	public void init_to_forward() {
		for (int algConfig = 0; algConfig < numberOfConfigurations ; algConfig++) {
			Function<T> alg = createForwards(algConfig);

			List<Case> testCases = createTestInputs();

			for( boolean sub : new boolean[]{false,true}) {
				for (Case test : testCases) {
					T tensor = tensorFactory.randomM(random,sub,test.minibatch,test.inputShape);

					List<T> parameters = new ArrayList<>();

					try {
						alg.initialize(test.inputShape);
						assertFalse( areExceptionsExpected(alg,test.inputShape));
					} catch( RuntimeException ignore ) {
						assertTrue( areExceptionsExpected(alg,test.inputShape));
						continue;
					}
					for( int[] s : alg.getParameterShapes() ) {
						parameters.add( tensorFactory.random(random,sub,s) );
					}
					T output = tensorFactory.randomM(random,sub,test.minibatch,alg.getOutputShape());

					alg.setParameters(parameters);
					alg.forward(tensor,output);
					// did it blow up?  If not it passes!
				}
			}
		}
	}

	/**
	 * Checks the shape returned by parameter and output shape getters
	 */
	@Test
	public void init_to_getters() {
		for (int algConfig = 0; algConfig < numberOfConfigurations; algConfig++) {
			Function<T> alg = createForwards(algConfig);
			List<Case> testCases = createTestInputs();

			for (Case testCase : testCases) {
				try {
					alg.initialize(testCase.inputShape);
					assertFalse( areExceptionsExpected(alg,testCase.inputShape));
				} catch( RuntimeException ignore ) {
					assertTrue( areExceptionsExpected(alg,testCase.inputShape));
					continue;
				}

				checkParameterShapes(testCase.inputShape, alg.getParameterShapes());
				checkOutputShapes(testCase.inputShape, alg.getOutputShape());
			}
		}
	}

	@Test
	public void checkTensorType() {
		for (int algConfig = 0; algConfig < numberOfConfigurations; algConfig++) {
			Function<T> alg = createForwards(algConfig);
			checkTensorType(alg.getTensorType());
		}
	}

	/**
	 * Checks to see if this is the exected tensor type
	 */
	protected abstract void checkTensorType( Class<T> type );

	/**
	 * Given the input, is this a valid parameter shape?
	 */
	protected abstract void checkParameterShapes(int []input , List<int[]> parameters );

	/**
	 * Given the input, is this a valid output shape?
	 */
	protected abstract void checkOutputShapes(int []input , int[] output);

}
