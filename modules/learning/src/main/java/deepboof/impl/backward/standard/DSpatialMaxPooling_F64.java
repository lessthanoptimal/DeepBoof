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

package deepboof.impl.backward.standard;

import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.forward.ConfigSpatial;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.Tensor_S32;

import java.util.List;

/**
 * Implementation of {@link DSpatialPadding2D_F64} for {@link Tensor_F64} that extends {@link DSpatialWindowChannel}.
 *
 * Comments:<br>
 *     dpadding is a 2D tensor of the spatial region only.  In the forwards pass the partial coordinate's index is
 *     saved and the batch + channel indexes are implicit saved in the output index tensor.
 *
 *
 * @author Peter Abeles
 */
public class DSpatialMaxPooling_F64 extends DSpatialWindowChannel<Tensor_F64,DSpatialPadding2D_F64> {

	// reference to dout and the input gradient
	Tensor_F64 dout;

	// contains the index of the maximum in the local padded image coordinate
	Tensor_S32 outputToPaddingIdx = new Tensor_S32();

	public DSpatialMaxPooling_F64(ConfigSpatial config, DSpatialPadding2D_F64 padding) {
		super(config, padding);
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {

		outputToPaddingIdx.reshape(output.getShape());
		forwardChannel(input, output);
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput,
							  List<Tensor_F64> gradientParameters) {
		this.dout = dout;
		gradientInput.zero();

		backwardsChannel(input, gradientInput);
	}

	@Override
	protected void backwardsAt_inner(Tensor_F64 input, int batch, int channel, int inY, int inX, int outY, int outX) {

		// The padded index is only for the spatial region
		int paddedIdx = outputToPaddingIdx.d[outputToPaddingIdx.idx(batch,channel,outY,outX)];
		dpadding.d[paddedIdx] += dout.get(batch,channel,outY,outX);
	}

	@Override
	protected void backwardsAt_border(DSpatialPadding2D_F64 padded, int batch, int channel, int padY, int padX, int outY, int outX) {
		int paddedIdx = outputToPaddingIdx.d[outputToPaddingIdx.idx(batch,channel,outY,outX)];
		dpadding.d[paddedIdx] += dout.get(batch,channel,outY,outX);
	}

	@Override
	protected void forwardAt_inner(Tensor_F64 input, int batch, int channel, int inY, int inX, int outY, int outX) {

		int inputIndexRow = input.idx(batch,channel,inY,inX);

		double max = -Double.MAX_VALUE;
		int maxX = -1, maxY = -1;

		for (int i = 0; i < HH; i++) {
			int inputIndex = inputIndexRow;

			for (int j = 0; j < WW; j++ , inputIndex++) {
				double value = input.d[inputIndex];
				if( value > max ) {
					max = value;
					maxX = j;
					maxY = i;
				}
			}

			inputIndexRow += W;
		}

		// save the results
		output.d[ output.idx(batch,channel,outY,outX) ] = max;

		// Compute index of maximum in padded image coordinates
		int padRow0 = padding.getPaddingRow0();
		int padCol0 = padding.getPaddingCol0();

		int index = (inY+maxY+padRow0)*Wp + (inX+maxX+padCol0);
		outputToPaddingIdx.d[ outputToPaddingIdx.idx(batch,channel,outY,outX) ] = index;
	}

	@Override
	protected void forwardAt_border(DSpatialPadding2D_F64 padded, int batch, int channel, int padY, int padX, int outY, int outX) {

		double max = -Double.MAX_VALUE;

		int maxX = -1, maxY = -1;

		for (int i = 0; i < HH; i++) {

			for (int j = 0; j < WW; j++ ) {
				double value = padded.get(batch,channel, padY +i, padX +j);
				if( value > max ) {
					max = value;
					maxX = j;
					maxY = i;
				}
			}
		}

		// Compute index of maximum in padded image coordinates
		int index = (padY+maxY)*Wp + (padX+maxX);

		// save the results
		output.d[ output.idx(batch,channel,outY,outX) ] = max;
		outputToPaddingIdx.d[ outputToPaddingIdx.idx(batch,channel,outY,outX) ] = index;
	}
}
