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
import deepboof.PaddingType;
import deepboof.factory.FactoryForwards;
import deepboof.tensors.Tensor_F32;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Base class for checks on spatial pooling
 *
 * @author Peter Abeles
 */
public abstract class ChecksForwardSpatialPooling_F32 extends ChecksForwardSpatialWindow_F32<ConfigSpatial> {

	public ChecksForwardSpatialPooling_F32() {
		numberOfConfigurations = 6;
	}

	@Override
	public Function<Tensor_F32> createForwards(int which) {
		int which0 = which%3;
		int which1 = which/3;

		config = new ConfigSpatial();
		configPadding = new ConfigPadding();
		if( which1 == 1)
			configPadding.type = PaddingType.CLIPPED;

		config.WW = 3;
		config.HH = 4;

		if( which0 == 1 ) {
			config.WW = 1;
			config.HH = 1;
		} else if( which0 == 2 ) {
			configPadding.x0 = 1;
			configPadding.x1 = 2;
			configPadding.y0 = 3;
			configPadding.y1 = 4;
		} else if( which0 != 0  ){
			throw new RuntimeException("Unexpected");
		}

		return createForwards(config,configPadding);
	}

	protected abstract Function<Tensor_F32> createForwards( ConfigSpatial configSpatial ,
															ConfigPadding configPadding );

	@Override
	public SpatialPadding2D_F32 createPadding(int which ) {
		return (SpatialPadding2D_F32)FactoryForwards.spatialPadding(configPadding,Tensor_F32.class);
	}

	@Override
	protected void checkParameterShapes(int[] input, List<int[]> parameters) {
		assertEquals(0,parameters.size());
	}

	@Override
	public int inputToOutputChannelCount(int numInput) {
		return numInput;
	}

}
