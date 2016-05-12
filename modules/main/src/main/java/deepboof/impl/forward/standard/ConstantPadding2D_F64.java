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

import deepboof.forward.ConfigPadding;
import deepboof.forward.ConstantPadding2D;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.tensors.Tensor_F64;

/**
 * Pads pixels outside the input image with a user specified constant value.
 *
 * @author Peter Abeles
 */
public class ConstantPadding2D_F64 extends SpatialPadding2D_F64
		implements ConstantPadding2D<Tensor_F64>
{
	// the value which the image is padded with
	double value;

	public ConstantPadding2D_F64(ConfigPadding config ) {
		super(config);
		switch( config.type ) {
			case ZERO: value = 0; break;
			case MAX_NEGATIVE: value = -Double.MAX_VALUE; break;
			case MAX_POSITIVE: value =  Double.MAX_VALUE; break;
			default: throw new IllegalArgumentException("Type doesn't specify a value");
		}
	}

	public ConstantPadding2D_F64(ConfigPadding config, double value ) {
		super(config);
		this.value = value;
	}

	@Override
	public double borderGet(int minibatch, int channel, int row, int col) {
		return value;
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}

	public /**/double getPaddingValue() {
		return value;
	}
}
