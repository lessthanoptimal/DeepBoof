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
import deepboof.forward.SpatialBatchNorm;

import java.util.List;

/**
 * <p>Interface of {@link SpatialBatchNorm Spatial Batch Normalization} for training networks.  Spatial batch norm
 * can be made to be functionally equivalent to regular batch norm by simply reordering each band so that all
 * the pixels inside are treated as one variable.  See {@link DFunctionBatchNorm} for additional details on
 * training method.</p>
 *
 * @author Peter Abeles
 */
public interface DSpatialBatchNorm<T extends Tensor<T>>
        extends DBatchNorm<T> {

    /**
     * <p>Performs batch normalization on spatial data.</p>
     *
     * <p>There is only a parameter tensor if {@link #hasGammaBeta()} returns true.  If true then
     * gamma, and beta are encoded in a single tensor in an interleaved fashion (gamma, beta).</p>
     * <pre>
     * Summary Table
     * -------------------------------------------------
     * Input   shape = (N, C, H,  W)
     * Output  shape = (N, C, H,  W)
     * Params  shape = (C, 2)
     * -------------------------------------------------
     * N   = Size of mini-batch
     * C   = Number of channels in input image
     * H   = Height of input image
     * W   = With of input image
     * </pre>
     *
     * @param input  Input tensor = (N,C,H,W)
     * @param output Output tensor = (N,C,H,W). Modified.
     */
    @Override
    void forward(T input, T output);

    /**
     * There are only parameters when gamma-beta is used.  See {@link #forward} for a description
     * parameter encoding.
     *
     * @param parameters Single tensor with shape (C, 2). Not modified.
     */
    @Override
    void setParameters(List<T> parameters);
}
