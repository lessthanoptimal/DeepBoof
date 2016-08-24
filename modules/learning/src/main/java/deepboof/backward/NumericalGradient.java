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

import deepboof.Function;
import deepboof.Tensor;

import java.util.List;

/**
 * <p>Given a {@link Function} implementations of this interface will compute the gradient of its
 * inputs and parameters.  Numerical differentiation is done using a symmetric sample, e.g.
 * {@code dx = [f(x+T)-f(x-T)]/T}</p>
 *
 * @author Peter Abeles
 */
public interface NumericalGradient<T extends Tensor<T>> {

	/**
	 * Overrides default settings for computing numerical gradient.
	 *
	 * @param T Sampling distance used for numerical differentiation
	 */
	void configure( double T );

	/**
	 * Sets the function which will be differentiated and other parameters.
	 * {@link Function#initialize(int...)}  should have already been called.
	 *
	 * @param function The function which is to be differentiated
	 */
	void setFunction(Function<T> function );

	/**
	 * Performs numerical differentiation to compute the gradients of input and parameters.  When numerical
	 * differentiation is being performed <code>input</code> and <code>parameters</code> will be modified
	 * and then returned to their original state.
	 *
	 * @param input The same input tensor which was passed in during the forward pass.
	 * @param parameters The same parameters which was passed in during the forward pass.
	 * @param dout Derivative of output, computed from next layer.
	 * @param gradientInput Storage for gradient of input
	 * @param gradientParameters  Storage for gradients of parameters
	 */
	void differentiate(T input, List<T> parameters, T dout ,
					   T gradientInput , List<T> gradientParameters );
}
