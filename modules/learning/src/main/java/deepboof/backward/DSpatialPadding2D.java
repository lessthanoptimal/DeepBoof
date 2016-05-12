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
import deepboof.forward.SpatialPadding2D;

/**
 * @author Peter Abeles
 */
public interface DSpatialPadding2D<T extends Tensor<T>> extends SpatialPadding2D<T>
{
	/**
	 * Compute the gradient of the input image from the gradient of the padded image for a specific mini-batch and
	 * channel of the input tensor.
	 *
	 * NOTE: gradientInput refers to the input in the forwards pass.  In the backwards pass (this function) it will
	 * be the output.
	 *
	 * @param gradientPadded (Input) Padded 2D image tensor for a specific channel.
	 * @param batch (Output) Index of the mini-batch in the input tensor
	 * @param channel (Output) Channel index in the input tensor
	 * @param gradientInput (Output) Output 4D spatial tensor.
	 */
	void backwardsChannel(T gradientPadded, int batch, int channel, T gradientInput);
}
