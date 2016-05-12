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
 * Performs convolutions across an input image with special kernels that have 'C' channels, one for each input image.
 *
 * @author Peter Abeles
 */
public interface SpatialConvolve2D<T extends Tensor<T>> extends Function<T> {

	/**
	 * Applies forward spacial convolution.  Each spacial convolution kernel is composed of a C 2D kernels that are
	 * applied to the input image.  The number of channels in the output image is dependent on the number of
	 * spacial kernels in this layer.
	 *
	 * <pre>
	 * Input   shape = (N, C, H,  W)
	 * Outputs shape = (N, F, H', W')
	 * Weight  shape = (F, C, HH, WW)
	 * Bias    shape = (F,)
	 * -------------------------------------------------
	 * N   = Size of mini-batch
	 * C   = Number of channels in input image
	 * H   = Height of input image
	 * W   = With of input image
	 * F   = Number of kernels or channels in output
	 * H'  = Height of output image. H' = 1 + (H + padY0 + padY1 - HH) / periodY
	 * W'  = Width of output image.  W' = 1 + (W + padX0 + padX1 - WW) / periodX
	 * HH  = Height of kernel
	 * WW  = Width of kernel
	 * </pre>
	 *
	 * @param input Tensor with the shape (N,C,H,W)
	 * @param output Output tensor (N, F, H', W')  Modified.
	 */
	@Override
	void forward(T input , T output );

	/**
	 * See {@link #forward} for a description of parameters.
	 *
	 * @param parameters Two tensors.  Weights = (F, C, HH, WW), bias = (F,)
	 */
	@Override
	void setParameters(List<T> parameters );

	/**
	 * Returns configuration of spacial parameters
	 * @return Copy of configuration
	 */
	ConfigConvolve2D getConfiguration();

	/**
	 * Returns the padding
	 */
	SpatialPadding2D<T> getPadding();
}
