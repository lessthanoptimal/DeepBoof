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
 * are no longer input parameters but are computed dynamically in the forwards pass.</p>
 *
 * NOTES:
 * <ul>
 *     <li>Variance is computed the unbiased formulation, i.e. divide by N-1 instead of N</li>
 * </ul>
 *
 * @author Peter Abeles
 */
public interface DFunctionBatchNorm<T extends Tensor<T>>
        extends DBatchNorm<T> {
    /**
     * <p>Applies batch normalization to each variable in the input.</p>
     *
     * <p>There is only a parameter tensor if {@link #hasGammaBeta()} returns true.  If true then
     * gamma, and beta are encoded in a single tensor in an interleaved fashion (gamma, beta).</p>
     *
     * <pre>
     * Summary Table
     * -------------------------------------------------
     * Input   shape = (N, d[i], ... , d[k])
     * Output  shape = (N, d[i], ... , d[k])
     * Params  shape = (d[i], ... , d[k], 2)
     * -------------------------------------------------
     * N    = Size of mini-batch
     * d[i] = length of a dimension
     * </pre>
     *
     * <p>NOTE: Interleaving is used in the parameters instead of multiple tensors to improve memory locality,
     * which reduces cache misses.</p>
     *
     * @param input Input tensor.  Tensor with a shape of (N, d[i], ... , d[k]), where N is mini-batch size
     * @param output Output tensor. Same shape as input tensor Modified.
     */
    @Override
    void forward(T input , T output );

    /**
     * See {@link #forward} for a description of parameters.
     *
     * @param parameters Variable tensor.  (d[i], ... , d[k], 2). Not modified.
     */
    @Override
    void setParameters(List<T> parameters );
}
