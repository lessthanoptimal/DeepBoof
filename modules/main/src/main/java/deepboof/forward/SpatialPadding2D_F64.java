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

import deepboof.impl.forward.standard.BaseSpatialPadding2D;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.VTensor_F64;

/**
 * Abstract class for F64 implementations of {@link BaseSpatialPadding2D}.  Provides
 * accessors for spatial tensors.
 *
 * @author Peter Abeles
 */
public abstract class SpatialPadding2D_F64 extends BaseSpatialPadding2D<Tensor_F64>
		implements VTensor_F64
{
	public SpatialPadding2D_F64(ConfigPadding config) {
		super(config);
	}

	/**
	 * Handles coordinates outside the input image
	 */
	public abstract double borderGet(int minibatch, int channel , int row , int col );

	/**
	 * {@inheritDoc}
	 */
	@Override
	public double get(int... coor) {
		if( coor.length == 4 )
			return get(coor[0],coor[1],coor[2],coor[3]);
		else
			throw new IllegalArgumentException("Expected 4-DOF spatial tensor");
	}

	/**
	 * Invalid accessor.  Only supports 4-DOF accessors.
	 */
	@Override
	public double get(int axis0) {
		throw new IllegalArgumentException("Expected 4-DOF spatial tensor");
	}

	/**
	 * Invalid accessor.  Only supports 4-DOF accessors.
	 */
	@Override
	public double get(int axis1, int axis0) {
		throw new IllegalArgumentException("Expected 4-DOF spatial tensor");
	}

	/**
	 * Invalid accessor.  Only supports 4-DOF accessors.
	 */
	@Override
	public double get(int axis2, int axis1, int axis0) {
		throw new IllegalArgumentException("Expected 4-DOF spatial tensor");
	}

	/**
	 * Returns the value of the virtual padded tensor at the specified coordinate.  The coordinate
	 * can be inside or outside the original image.
	 *
	 * @param minibatch mini-batch number
	 * @param channel channel in spatial tensor
	 * @param row Row in padded coordinates
	 * @param col Column in padded coordinates
	 * @return Value of padded tensor
	 */
	@Override
	public double get( int minibatch , int channel , int row , int col ) {
		if( row < ROW0 || row >= ROW1 || col < COL0 || col >= COL1 ) {
			return borderGet(minibatch,channel,row,col);
		} else {
			return input.d[input.idx(minibatch, channel, row-ROW0, col-COL0)];
		}
	}

	/**
	 * Invalid accessor.  Only supports 4-DOF accessors.
	 */
	@Override
	public double get(int axis4, int axis3, int axis2, int axis1, int axis0) {
		throw new IllegalArgumentException("Expected 4-DOF spatial tensor");
	}

	@Override
	public Class getDataType() {
		return double.class;
	}
}
