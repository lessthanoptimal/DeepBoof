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
 * Applies a linear (or affine) equation to input array.  This is a matrix multiplication operation that is
 * performed between the input vector and their respective weights plus bias.
 * <pre>
 * y = W*x + b
 * </pre>
 * where y is the output, W are the weights, x is the input, and b is the bias.
 *
 * @author Peter Abeles
 */
public interface FunctionLinear<T extends Tensor> extends Function<T> {
	/**
	 * <p>
	 * Input tensor shape =(N, d[i], ... , d[K])<br>
	 * where N is the number of stacked inputs, d[i] is the number of inputs along dimension i, and K is
	 * the dimension of the input space.
	 * </p>
	 *
	 * <p>
	 * Tensor Parameters:<br>
	 * Weight matrix shape=(M,D)<br>
	 * bias shape=(M)<br>
	 * where D is the total number of inputs (Product d[i]) and M is the number of outputs.
	 * </p>
	 *
	 * <p>NOTE: The tensor parameters' shape has been selected to minimize cache misses during matrix multiplication.</p>
	 *
	 * @param input Tensor with a shape of (N, d[i], ... , d[k])
	 * @param output Output tensor with shape (N,M). Modified.
	 */
	void forward(T input , T output );

	/**
	 * See {@link #forward} for a description of parameters.
	 *
	 * @param parameters Weight and bias tensors with shapes (M, D), and (M,), respectively.
	 */
	@Override
	void setParameters(List<T> parameters );

	/**
	 * Returns the number of output elements. This is the variable M.
	 * @return Number of output elements.
	 */
	int getNumberOfOutputs();
}
