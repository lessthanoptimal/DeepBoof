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
import deepboof.forward.FunctionBatchNorm;

import java.util.List;

/**
 * <p>Implementation of {@link FunctionBatchNorm Batch Normalization} for training networks.  This has distinctly
 * different behavior from forward only implementations.  In this learning implementation, statistics of
 * the input parameters are recomputed every time {@link #forward} is invoked.  While for the forward only
 * implementation those statistics are known already and not recomputed.</p>
 *
 * <p>The above described change in behavior also changes how parameters are specified.  mean and variance
 * are no longer input parameters but are computed dynamically.</p>
 *
 * @author Peter Abeles
 */
public interface DFunctionBatchNorm<T extends Tensor<T>> extends FunctionBatchNorm<T>, DFunction<T> {
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
     * M    = Number of parameters.  0 or 2 if gamma-beta is being used.
     *       in order of: gamma, beta
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
     * @param parameters Variable tensor.  (d[i], ... , d[k], M), where M is 0 or 2. Not modified.
     */
    @Override
    void setParameters(List<T> parameters );

    /**
     * Returns the most recently computed mean.
     *
     * @param output Storage for mean tensor. Is reshaped. If null a new instance will be decalred
     *
     */
    T getMean( T output );

    /**
     * Returns the most recently computed variance.  This will be the actual variance not something that has been
     * adjusted by adding EPS to it.
     *
     * @param output Storage for variance tensor. Is reshaped. If null a new instance will be decalred
     */
    T getVariance( T output );
}
