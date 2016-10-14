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

import deepboof.misc.Configuration;

/**
 * Configuration for 2D convolution.  See {@link SpatialConvolve2D} for a more detailed description of variable names.
 *
 * @author Peter Abeles
 */
public class ConfigConvolve2D extends ConfigSpatial implements Configuration {

	/**
	 * Number of kernels
	 */
	public int F;

	/**
	 * Makes sure valid configurations are set
	 */
	@Override
	public void checkValidity() {
		super.checkValidity();

		if( F <= 0 )
			throw new IllegalArgumentException("F must be > 0");
	}

	/**
	 * Number of kernels which will be convolved across the input.  This specifies the number of output channels.
	 *
	 * @return Number of kernels which are convolved across the input, a.k.a. output channels.
	 */
	public int getTotalKernels() {
		return F;
	}

	@Override
	public ConfigConvolve2D clone() {
		ConfigConvolve2D c = new ConfigConvolve2D();

		c.WW = WW;
		c.HH = HH;
		c.F = F;
		c.periodX = periodX;
		c.periodY = periodY;

		return c;
	}
}
