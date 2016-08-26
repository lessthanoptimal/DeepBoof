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
 * <p>Implementation of a forward only Batch Normalization.  It applies a previously computed linear transform
 * which will ensure that the training data will have an output with zero mean and standard deviation (stdev) of
 * one.  The optional gamma and beta transform can also be applied.</p>
 *
 * <p>See {@link BatchNorm} for a general discussion of Batch Normalization</p>
 *
 * @author Peter Abeles
 */
public interface FunctionBatchNorm<T extends Tensor<T>> extends Function<T>, BatchNorm {

	/**
	 * <p>Applies batch normalization to each variable in the input.</p>
	 *
	 * <p>Either two or four variables are stored in the parameter tensor as interleaved variables.  If
	 * {@link #hasGammaBeta()} returns true then mean, variance, gamma, and beta are saved.  Otherwise just
	 * mean, and variance are saved. These are also the order in which variables are interleaved together.</p>
	 *
	 * <pre>
	 * Summary Table
	 * -------------------------------------------------
	 * Input   shape = (N, d[i], ... , d[k])
	 * Output  shape = (N, d[i], ... , d[k])
	 * Params  shape = (d[i], ... , d[k], M)
	 * -------------------------------------------------
	 * N    = Size of mini-batch
	 * d[i] = length of a dimension
	 * M    = Number of parameters.  2 or 4 if gamma-beta is being used.
	 *       in order of: mean, variance  OR mean, variance, gamma, beta
	 * </pre>
	 *
	 * <p>NOTE: Interleaving is used instead of multiple tensors to improve memory locality, which reduces cache
	 * misses.</p>
	 *
	 * @param input Input tensor.  Tensor with a shape of (N, d[i], ... , d[k]), where N is mini-batch size
	 * @param output Output tensor. Same shape as input tensor Modified.
	 */
	@Override
	void forward(T input , T output );

	/**
	 * See {@link #forward} for a description of parameters.
	 *
	 * @param parameters Variable tensor.  (d[i], ... , d[k], M), where M is 2 or 4. Not modified.
	 */
	@Override
	void setParameters(List<T> parameters );


}
