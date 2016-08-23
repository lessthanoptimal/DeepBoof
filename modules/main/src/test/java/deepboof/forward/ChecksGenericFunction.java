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

import deepboof.Tensor;
import deepboof.misc.TensorFactory;

import java.util.List;
import java.util.Random;

/**
 * provides skeleton for performing very basic tests of the contract for Function
 *
 * @author Peter Abeles
 */
public abstract class ChecksGenericFunction<T extends Tensor<T>> {

	protected Random random = new Random(234);

	protected TensorFactory<T> tensorFactory;

	/**
	 * Configure of distinctive configurations.  Each configuration might have different shapes for parameters.
	 */
	protected int numberOfConfigurations=1;

	public ChecksGenericFunction(int numberOfConfigurations) {
		this.numberOfConfigurations = numberOfConfigurations;
	}

	public ChecksGenericFunction() {
	}

	public abstract List<Case> createTestInputs();

	public static class Case {
		// shape of input tensor
		public int[] inputShape;
		// number of minibatches
		public int minibatch = 3;

		public Case( int ...shape ) {
			this.inputShape = shape;
		}
	}

	public static Case minione( int ...shape ) {
		Case c = new Case(shape);
		c.minibatch = 1;
		return c;
	}
}
