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

import java.util.List;

/**
 * Rectified Linear Unit (ReLU) activation function.  Used in [1] as an alternative to tanh with the claim
 * that it has much better convergence.
 * <pre>
 * f(x) = 0  if x &lt; 0
 *        x     x &ge; 0
 * </pre>
 * <p> [1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional
 * neural networks." Advances in neural information processing systems. 2012.
 * </p>
 *
 * @author Peter Abeles
 */
public interface ActivationReLU<T extends Tensor> extends Function<T> {

	/**
	 * Can skip.  No parameters required.
	 *
	 * @param parameters No parameters required
	 */
	@Override
	void setParameters( List<T> parameters );

	/**
	 * Applies the ReLU operator to each element in the input tensor and saves the results
	 * in the output tensor.  Any shape is allowed.
	 *
	 * @param input Input to the function.  Any shape.
	 * @param output Output tensor. Same shape as input. Modified.
	 */
	@Override
	void forward(T input ,  T output );
}
