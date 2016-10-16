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

package deepboof.impl.forward.standard;

import deepboof.Function;
import deepboof.factory.FactoryForwards;
import deepboof.forward.ChecksForwardSpatialAveragePooling_F32;
import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialPadding2D_F32;
import deepboof.tensors.Tensor_F32;

/**
 * @author Peter Abeles
 */
public class TestSpatialAveragePooling_F32 extends ChecksForwardSpatialAveragePooling_F32 {

	@Override
	protected Function<Tensor_F32> createForwards(ConfigSpatial configSpatial,
												  ConfigPadding configPadding) {

		SpatialPadding2D_F32 padding = (SpatialPadding2D_F32)
				FactoryForwards.spatialPadding(configPadding,Tensor_F32.class);

		return new SpatialAveragePooling_F32(configSpatial,padding);
	}
}
