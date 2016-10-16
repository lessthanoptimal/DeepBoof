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

package deepboof.factory;

import deepboof.Tensor;
import deepboof.forward.ConfigPadding;
import deepboof.impl.forward.standard.*;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class FactoryForwards {

	public static <T extends Tensor<T>> BaseSpatialPadding2D<T> spatialPadding(ConfigPadding config , Class<T> type ) {
		if( type == Tensor_F64.class ) {
			switch( config.type ) {
				case ZERO:
				case MAX_NEGATIVE:
					return (BaseSpatialPadding2D<T>)new ConstantPadding2D_F64(config);
				case CLIPPED:
					return (BaseSpatialPadding2D<T>)new ClippedPadding2D_F64(config);
			}
		} else if( type == Tensor_F32.class ) {
			switch( config.type ) {
				case ZERO:
				case MAX_NEGATIVE:
					return (BaseSpatialPadding2D<T>)new ConstantPadding2D_F32(config);
				case CLIPPED:
					return (BaseSpatialPadding2D<T>)new ClippedPadding2D_F32(config);
			}
		}
		throw new IllegalArgumentException("Unsupported");
	}
}
