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

import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialAveragePooling;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link SpatialAveragePooling} for {@link Tensor_F64}.
 *
 * @author Peter Abeles
 */
public class SpatialAveragePooling_F64
		extends SpatialWindowChannel<Tensor_F64,SpatialPadding2D_F64>
		implements SpatialAveragePooling<Tensor_F64>
{
	// number of elements inside the pooling region
	protected double poolingSize;

	public SpatialAveragePooling_F64(ConfigSpatial config , SpatialPadding2D_F64 padding ) {
		super(config, padding);
	}

	@Override
	public void _initialize() {
		super._initialize();
		if( shapeInput.length != 3 )
			throw new IllegalArgumentException("Expected 3D spatial tensor");

		shapeOutput = shapeInput.clone();
		shapeOutput[1] = Ho;
		shapeOutput[2] = Wo;

		poolingSize = WW*HH;
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		forwardChannel(input, output);
	}

	@Override
	protected void forwardAt_inner(Tensor_F64 input, int batch, int channel, int inY, int inX, int outY, int outX) {

		int inputIndexRow = input.idx(batch,channel,inY,inX);

		double sum = 0;

		for (int j = 0; j < HH; j++) {
			int inputIndex = inputIndexRow;

			for (int i = 0; i < WW; i++ ) {
				sum += input.d[inputIndex++];
			}

			inputIndexRow += W;
		}

		// save the results
		output.d[ output.idx(batch,channel,outY,outX) ] = sum/poolingSize;
	}

	@Override
	protected void forwardAt_border(SpatialPadding2D_F64 padded, int batch, int channel, int padY, int padX, int outY, int outX) {

		int row0 = padY;
		int row1 = padY + HH;
		row0 += padded.getClippingOffsetRow(row0);
		row1 += padded.getClippingOffsetRow(row1);

		int col0 = padX;
		int col1 = padX + WW;
		col0 += padded.getClippingOffsetCol(col0);
		col1 += padded.getClippingOffsetCol(col1);

		double sum = 0;

		for (int j = row0; j < row1; j++) {
			for (int i = col0; i < col1; i++ ) {
				sum += padded.get(batch,channel, j, i);
			}
		}

		// save the results
		output.d[ output.idx(batch,channel,outY,outX) ] = sum/((row1-row0)*(col1-col0));
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}

	@Override
	public ConfigSpatial getConfiguration() {
		return config;
	}
}

