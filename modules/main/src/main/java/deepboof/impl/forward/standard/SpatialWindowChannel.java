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

import deepboof.Tensor;
import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialPadding2D;

/**
 * Implementation of {@link BaseSpatialWindow} which processes the spatial tensor is processed in
 * BCHW (mini-batch, channel, height, width) order
 *
 * @author Peter Abeles
 */
public abstract class SpatialWindowChannel
		<T extends Tensor<T>,VT extends SpatialPadding2D<T>>
		extends BaseSpatialWindow<T,VT>
{
	// reference to output tensor
	protected T output;

	public SpatialWindowChannel(ConfigSpatial config, VT padding) {
		super(config, padding);
	}

	protected void forwardChannel(T input, T output) {
		this.output = output;
		padding.setInput(input);

		// extract constants which describe the convolution from inputs and parameters
		N = input.length(0);

		int paddingX0 = padding.getPaddingCol0();
		int paddingY0 = padding.getPaddingRow0();

		// lower and upper extends for where the input image is inside of the padded image
		int outC0 = innerLowerExtent(config.periodX,paddingX0);
		int outC1 = innerUpperExtent(config.WW,config.periodX,paddingX0,W);
		int outR0 = innerLowerExtent(config.periodY,paddingY0);
		int outR1 = innerUpperExtent(config.HH,config.periodY,paddingY0,H);

		if(isEntirelyBorder(outR0, outC0)) {
			// Handle the case where the entire output touches the border

			for (int batchIndex = 0; batchIndex < N; batchIndex++) {
				for( int channel = 0; channel < C; channel++ ) {
					forwardBorder(batchIndex,channel, 0, 0, Ho, Wo);
				}
			}
		} else {
			// Handle the case where there is at least one inner region which doesn't touch the border

			for (int batchIndex = 0; batchIndex < N; batchIndex++) {
				for( int channel = 0; channel < C; channel++ ) {

					// do the inner region first, which can be processed efficiently
					for (int outRow = outR0; outRow < outR1; outRow++) {
						int inputRow = outRow * config.periodY - paddingY0;

						for (int outCol = outC0; outCol < outC1; outCol++) {
							int inputCol = outCol * config.periodX - paddingX0;

							forwardAt_inner(input, batchIndex, channel, inputRow, inputCol, outRow, outCol);
						}
					}
					// Process the borders, top, bottom, left, right
					forwardBorder(batchIndex,channel, 0, 0, outR0, Wo);
					forwardBorder(batchIndex,channel, outR1, 0, Ho, Wo);
					forwardBorder(batchIndex,channel, outR0, 0, outR1, outC0);
					forwardBorder(batchIndex,channel, outR0, outC1, outR1, Wo);

				}
			}
		}
	}

	/**
	 * Processes along the spatial tensor's border using the padded virtual tensor.
	 *
	 * @param batchIndex which mini-batch
	 * @param row0 Lower extent along rows, inclusive.  output coordinates
	 * @param col0 Lower extent along columns, inclusive.  output coordinates
	 * @param row1 Upper extent along rows, exclusive.  output coordinates
	 * @param col1 Upper extent along columns, exclusive.  output coordinates
	 */
	private void forwardBorder(int batchIndex , int channel , int row0, int col0, int row1, int col1 ) {
		for (int outRow = row0; outRow < row1; outRow++) {
			int padRow = outRow*config.periodY;
			for (int outCol = col0; outCol < col1; outCol++) {
				int padCol = outCol*config.periodX;

				forwardAt_border(padding, batchIndex, channel, padRow, padCol, outRow, outCol);
			}
		}
	}

	/**
	 * Applies the operations at the specified window and stores the results at the specified output
	 * coordinate.
	 *
	 * @param input Input spatial tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param channel Channel
	 * @param inY y-axis lower extent, in input coordinates
	 * @param inX x-axis lower extent, in input coordinates
	 * @param outY y-axis output coordinates
	 * @param outX x-axis output coordinates
	 */
	protected abstract void forwardAt_inner(T input, int batch, int channel,
											int inY, int inX, int outY, int outX);

	/**
	 * Applies the operations at the specified window and stores the results at the specified output
	 * coordinate.  For virtual tensor
	 *
	 * @param padded Input spatial virtual tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param channel Channel
	 * @param padY y-axis lower extent, in padded coordinates
	 * @param padX x-axis lower extent, in padded coordinates
	 * @param outY y-axis output coordinates
	 * @param outX x-axis output coordinates
	 */
	protected abstract void forwardAt_border(VT padded, int batch, int channel,
											 int padY, int padX, int outY, int outX);

}
