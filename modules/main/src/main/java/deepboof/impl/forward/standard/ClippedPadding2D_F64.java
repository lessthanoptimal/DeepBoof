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

import deepboof.PaddingType;
import deepboof.forward.ClippedPadding2D;
import deepboof.forward.ConfigPadding;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.tensors.Tensor_F64;

/**
 * Implementation of {@link ConstantPadding2D_F64}.
 *
 * @author Peter Abeles
 */
public class ClippedPadding2D_F64 extends SpatialPadding2D_F64
		implements ClippedPadding2D<Tensor_F64>
{
	public ClippedPadding2D_F64(ConfigPadding config ) {
		super(config);
		if( config.type != PaddingType.CLIPPED) {
			throw new IllegalArgumentException("configuraiton isn't for clipped padding");
		}
	}

	@Override
	public double borderGet(int minibatch, int channel, int row, int col) {
		throw new RuntimeException("The border is clipped and this function should never be called");
	}

	@Override
	public int getClippingOffsetRow(int paddedRow) {
		if( paddedRow < ROW0)
			return ROW0 - paddedRow;
		else if( paddedRow > ROW1 )
			return ROW1 - paddedRow;
		return 0;
	}

	@Override
	public int getClippingOffsetCol(int paddedCol) {
		if( paddedCol < COL0)
			return COL0 - paddedCol;
		else if( paddedCol > COL1 )
			return COL1 - paddedCol;
		return 0;
	}

	@Override
	public boolean isClipped() {
		return true;
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
