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
import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.impl.backward.standard.DConstantPadding2D_F64;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
// TODO specify padding and kernel size in tests cases
public abstract class ChecksDerivativeSpatial<T extends Tensor<T>>
	extends ChecksDerivative<T>
{
	protected final ConfigSpatial configSpatial = new ConfigSpatial();

	// window size
	protected final int WW = 3;
	protected final int YY = 4;

	// padding around the input spatial tensor
	protected final int padY0 = 3;
	protected final int padY1 = 3;
	protected final int padX0 = 2;
	protected final int padX1 = 2;

	public ChecksDerivativeSpatial() {
		configSpatial.WW = WW;
		configSpatial.HH = YY;
	}

	protected DSpatialPadding2D_F64 createPadding() {
		ConfigPadding config = new ConfigPadding();
		config.y0 = padY0;
		config.x0 = padX0;
		config.y1 = padY1;
		config.x1 = padX1;

		return new DConstantPadding2D_F64(config);
	}


	@Override
	public List<Case> createTestInputs() {
		List<Case> valid = new ArrayList<>();

		// Just one channel for simplicity
		valid.add( minione(1,2,2) );

		int numChannels = 3;
		// Every window will go outside the input's border
		valid.add( new Case(numChannels,2,2));

		// Some will be border and some will not
		valid.add( new Case(numChannels,7,8));

		return valid;
	}
}
