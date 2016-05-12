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

/**
 * Multiplies each element in a tensor by the same value.
 *
 * @author Peter Abeles
 */
public interface FunctionElementWiseMult<T extends Tensor> extends Function<T>
{
	/**
	 * <p>Performs scalar multiplication on each element in the input tensor.</p>
	 *
	 * <pre>
	 * Summary Table
	 * -------------------------------------------------
	 * Input   shape = (N, d[i], ... , d[k])
	 * -------------------------------------------------
	 * N    = Size of mini-batch
	 * d[i] = length of a dimension
	 * </pre>
	 *
	 * @param input Input tensor of any shape.
	 * @param output Output tensor. Same shape as input tensor.
	 */
	@Override
	void forward(T input , T output );
}
