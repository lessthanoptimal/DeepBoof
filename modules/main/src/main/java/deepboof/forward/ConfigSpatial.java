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
 * Common configuration for many spatial functions
 *
 * @author Peter Abeles
 */
public class ConfigSpatial implements Configuration {

	/**
	 * Sample period. One is default.
	 */
	public int periodX=1,periodY=1;

	/**
	 * Window width
	 */
	public int WW;
	/**
	 * Window height
	 */
	public int HH;

	@Override
	public void checkValidity() {

		if( periodX <= 0 )
			throw new IllegalArgumentException("periodX must be > 0");
		if( periodY <= 0 )
			throw new IllegalArgumentException("periodY must be > 0");

		if( WW <= 0 )
			throw new IllegalArgumentException("Pooling region width must be more than zero");
		if( HH <= 0 )
			throw new IllegalArgumentException("Pooling region height must be more than zero");
	}

	/**
	 * The period at which the kernel samples the input image along the x-axis
	 *
	 * @return sample period along x-axis in pixels
	 */
	public int getPeriodX() {
		return periodX;
	}

	/**
	 * The period at which the kernel samples the input image along the y-axis
	 *
	 * @return sample period along y-axis in pixels
	 */
	public int getPeriodY() {
		return periodY;
	}
}
