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
 * Interface for computing the gradient of a padded spatial tensor.  Spatial tensors have the shape of (N, C, H,  W)
 * where N is the number of mini-batches, C channels, H height and W width.
 *
 * @author Peter Abeles
 */
public interface DSpatialPadding2D<T extends Tensor<T>> extends SpatialPadding2D<T>
{
	/**
	 * <p>Compute the gradient of the input image from the gradient of the padded image for a specific mini-batch and
	 * channel of the input tensor.</p>
	 *
	 * @param gradientPadded (Input) Gradient of padded image at specific mini-batch and channel.  (H,  W)
	 * @param batch (Input) mini-batch.
	 * @param channel (Input) Channel.
	 * @param gradientInput (Output) Output 4D spatial tensor.  Only elements used to compute the channel are modified.
	 *                      (N, C, H,  W)
	 */
	void backwardsChannel(T gradientPadded, int batch, int channel, T gradientInput);

	/**
	 * <p>Compute the gradient of the input image from the gradient of the padded image for a specific mini-batch and
	 * channel of the input tensor.</p>
	 *
	 * @param gradientPadded (Input) Gradient of padded image at specific mini-batch.  (C, H,  W)
	 * @param batch (Input) mini-batch.
	 * @param gradientInput (Output) Output 4D spatial tensor.  Only elements used to compute the image are modified.
	 *                      (N, C, H,  W)
	 */
	void backwardsImage(T gradientPadded, int batch, T gradientInput);
}
