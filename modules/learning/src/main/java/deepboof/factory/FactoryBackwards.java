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
import deepboof.backward.DSpatialPadding2D;
import deepboof.backward.NumericalGradient;
import deepboof.forward.ConfigPadding;
import deepboof.impl.backward.standard.DClippedPadding2D_F64;
import deepboof.impl.backward.standard.DConstantPadding2D_F64;
import deepboof.impl.backward.standard.NumericalGradient_F64;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
public class FactoryBackwards<T extends Tensor<T>> {

	Class<T> tensorType;

	public FactoryBackwards(Class<T> tensorType) {
		this.tensorType = tensorType;
	}

	public NumericalGradient<T> createNumericalGradient() {
		if( tensorType == Tensor_F64.class )
			return (NumericalGradient)new NumericalGradient_F64();
		else
			throw new IllegalArgumentException("Unknown");
	}

	public <P extends DSpatialPadding2D<T>> P spatialPadding(ConfigPadding config) {
		if( tensorType == Tensor_F64.class ) {
			switch( config.type ) {
				case ZERO:
				case MAX_NEGATIVE:
					return (P)new DConstantPadding2D_F64(config);

				case CLIPPED:
					return (P)new DClippedPadding2D_F64(config);
			}
		} else if( tensorType == Tensor_F32.class ) {
//			switch( config.type ) {
//				case ZERO:
//				case MAX_NEGATIVE:
//					return (P)new DConstantPadding2D_F64(config);
//			}
		}
		throw new IllegalArgumentException("Unsupported");
	}
}
