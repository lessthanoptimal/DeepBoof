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
 * <p>Spatial {@link FunctionBatchNorm Batch Normalization} seeks to maintain the convolutional property, "that
 * different elements of the same feature map, at different locations, are normalized in the same way." [1]
 * Thus the input tensor (N,C,H,W) is "reshaped" such that it is (N*H*W,C) and it's treated like a mini-batch
 * with N*H*W elements.</p>
 *
 * <p>See {@link BatchNorm} for a general discussion of Batch Normalization</p>
 *
 * @author Peter Abeles
 */
public interface SpatialBatchNorm<T extends Tensor<T>> extends Function<T>, BatchNorm {
	/**
	 * Performs batch norm on spatial data.
	 *
	 * <pre>
	 * Summary Table
	 * -------------------------------------------------
	 * Input   shape = (N, C, H,  W)
	 * Output  shape = (N, C, H,  W)
	 * Params  shape = (C, M)
	 * -------------------------------------------------
	 * N   = Size of mini-batch
	 * C   = Number of channels in input image
	 * H   = Height of input image
	 * W   = With of input image
	 * M   = Number of parameters.  2 or 4 if gamma-beta is being used.
	 *       in order of: mean, stdev  OR mean, stdev, gamma, beta
	 * </pre>
	 *
	 * @param input Input tensor = (N,C,H,W)
	 * @param output Output tensor = (N,C,H,W). Modified.
	 */
	@Override
	void forward(T input , T output );

	/**
	 * See {@link #forward} for a description of parameters.
	 *
	 * @param parameters Variable tensor.  (C, M), where M is 2 or 4. Not modified.
	 */
	@Override
	void setParameters(List<T> parameters );
}
