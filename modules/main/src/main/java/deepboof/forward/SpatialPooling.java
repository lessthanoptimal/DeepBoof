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
 * Spatial pooling down samples the input spatial tensors by finding a representative value inside
 * each pooling region.  The intent is to reduce the number of variables while maintaining much of the original
 * information.  Pooling is specified by the region's size (poolWidth, poolHeight) and the sampling
 * period (periodX, periodY), and padding parameters.
 *
 * Processing Steps:<br>
 * <ol>
 *   <li>Apply spatial padding</li>
 *   <li>Apply spatial pooling to padded image</li>
 * </ol>
 *
 * <p>Notes</p>
 * <ul>
 *   <li>The first region has it's lower extent at the spatial region's lower extent.</li>
 *   <li>Sampling is done in a row-major ordering, e.g. columns then rows</li>
 *   <li>If a region extends outside the image plus padding it's ignore</li>
 * </ul>
 *
 * @see SpatialPadding2D
 * @see SpatialMaxPooling
 * @see SpatialAveragePooling
 *
 * @author Peter Abeles
 */
public interface SpatialPooling<T extends Tensor> extends Function<T> {

	/**
	 * Processes a spatial tensor.
	 *
	 * <pre>
	 * N  = number of mini-batch images
	 * C  = number of channels in each image
	 * H  = height of input image
	 * W  = width of input image
	 * Hp = height of input image + padding
	 * Wp = width of input image + padding
	 *
	 * Shape of output spacial tensor:
	 *
	 * H' = 1 + (Hp - poolHeight) / periodY
	 * W' = 1 + (Wp - poolWidth ) / periodX
	 * </pre>
	 *
	 * @param input Input spacial tensor = (N, C, H, W)
	 * @param output Output spatial tensor = (N, C, H', W')
	 */
	@Override
	void forward(T input, T output);

	/**
	 * Can skip.  No parameters required.
	 *
	 * @param parameters No parameters required
	 */
	@Override
	void setParameters(List<T> parameters);

	/**
	 * Returns pooling configuration
	 * @return configuration
	 */
	ConfigSpatial getConfiguration();
}
