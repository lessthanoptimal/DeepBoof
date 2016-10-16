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

package deepboof.impl.forward.torch;

import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.impl.forward.standard.SpatialAveragePooling_F64;

/**
 * Torch handles the border in a weird way.  It does not adjust the denominator when the number of pixels
 * used to compute the average is reduced.
 *
 * @author Peter Abeles
 */
public class TorchSpatialAveragePooling_F64 extends SpatialAveragePooling_F64 {
	public TorchSpatialAveragePooling_F64(ConfigSpatial config, SpatialPadding2D_F64 padding) {
		super(config, padding);
	}

	@Override
	protected void forwardAt_border(SpatialPadding2D_F64 padded,
									int batch, int channel, int padY, int padX, int outY, int outX) {
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
		output.d[ output.idx(batch,channel,outY,outX) ] = sum/poolingSize;
	}
}
