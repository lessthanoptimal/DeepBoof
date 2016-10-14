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

import deepboof.PaddingType;
import deepboof.misc.Configuration;

/**
 * Configuration for spatial padding.
 *
 * @author Peter Abeles
 */
public class ConfigPadding implements Configuration {
	/**
	 * Padding added to lower extent along x and y axis. Zero is default
	 */
	public int x0, y0;

	/**
	 * Padding added to upper extent along x and y axis.  Zero is default
	 */
	public int x1, y1;

	/**
	 * Type of padding added to input images
	 */
	public PaddingType type = PaddingType.ZERO;

	@Override
	public void checkValidity() {
		if( x0 < 0 )
			throw new IllegalArgumentException("padX0 must be >= 0");
		if( y0 < 0 )
			throw new IllegalArgumentException("padY0 must be >= 0");
		if( x1 < 0 )
			throw new IllegalArgumentException("padX1 must be >= 0");
		if( y1 < 0 )
			throw new IllegalArgumentException("padY1 must be >= 0");
	}

	/**
	 * Padding applied to input data along the lower extent of X axis
	 * @return padding in pixels
	 */
	public int getX0() {
		return x0;
	}

	/**
	 * Padding applies to input data along the lower extent of Y axis
	 * @return padding in pixels
	 */
	public int getY0() {
		return y0;
	}

	/**
	 * Padding applied to input data along the upper extent of X axis
	 * @return padding in pixels
	 */
	public int getX1() {
		return x1;
	}

	/**
	 * Padding applies to input data along the upper extent of Y axis
	 * @return padding in pixels
	 */
	public int getY1() {
		return y1;
	}

	@Override
	public ConfigPadding clone() {
		ConfigPadding ret = new ConfigPadding();

		ret.x0 = x0;
		ret.x1 = x1;
		ret.y0 = y0;
		ret.y1 = y1;
		ret.type = type;

		return ret;
	}
}
