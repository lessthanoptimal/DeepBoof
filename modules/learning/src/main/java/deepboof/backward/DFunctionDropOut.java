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

import deepboof.DFunction;
import deepboof.Tensor;

/**
 * Drop out is a technique introduced by [1] for regularizing a network and helps prevents over fitting.  It works
 * by randomly selecting neurons and forces them to be off.  The chance of a neuron being turned off is specified
 * by the drop rate.  It's behavior is different when in learning or evaluation mode.  In learning mode it will
 * decide if a neuron is dropped using a probability of drop_rate*100, drop_rate is 0 to 1.0, inclusive.
 * In evaluation mode it scales each input by 1.0 - drop_rate.
 *
 * <p>
 * [1] Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
 * </p>
 *
 * @author Peter Abeles
 */
public interface DFunctionDropOut<T extends Tensor<T>> extends DFunction<T> {

	/**
	 * Returns a number from 0 to 1 indicating the likelihood of a neuron being dropped.  0 = 0% change
	 * and 1 = 100% chance
	 *
	 * @return drop rate
	 */
	double getDropRate();
}
