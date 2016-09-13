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

import deepboof.DFunction;
import deepboof.Tensor;
import deepboof.backward.DSpatialPadding2D;
import deepboof.forward.ConfigSpatial;
import deepboof.impl.forward.standard.SpatialWindowChannel;
import deepboof.misc.TensorFactory;
import deepboof.misc.TensorOps;

import java.util.List;

/**
 * Backwards functions for operations which convolve a window across the input spatial tensor and
 * process the image in a BCHW (batch, channel, (row, column)) order, e.g. one channel at a time.
 *
 * @author Peter Abeles
 */
public abstract class DSpatialWindowChannel<T extends Tensor<T>, P extends DSpatialPadding2D<T>>
		extends SpatialWindowChannel<T,P> implements DFunction<T>
{
	// Toggle indicating if it's in learning mode or not
	protected boolean learningMode = false;

	// storage for padded image gradient.  This is a 2D tensor
	protected T dpadding;

	public DSpatialWindowChannel(ConfigSpatial config, P padding) {
		super(config, padding);

		dpadding = new TensorFactory<T>(padding.getTensorType()).create();
	}

	@Override
	public void backwards(T input, T dout, T gradientInput, List<T> gradientParameters) {

		if( shapeInput == null )
			throw new IllegalArgumentException("Must initialize first!");

		TensorOps.checkShape("input",-1,shapeInput,input.getShape(),true);

		TensorOps.checkShape("dout", -1, shapeOutput, dout.getShape(),true);
		TensorOps.checkShape("gradientInput",-1, shapeInput,gradientInput.getShape(),true);
		TensorOps.checkShape("gradientParameters", shapeParameters,(List)gradientParameters,false);

		_backwards(input,dout,gradientInput,gradientParameters);
	}

	protected abstract void _backwards(T input, T dout,  T gradientInput, List<T> gradientParameters);

	/**
	 * Convolve window across 'input' spatial tensor and compute its gradient
	 *
	 * @param input Input spatial tensor
	 * @param gradientInput Storage for input's gradient
	 */
	public void backwardsChannel(T input, T gradientInput ) {
		padding.setInput(input);

		// only need to do the spatial component for 1 channel
		int[] paddingShape = padding.getShape();
		dpadding.reshape(paddingShape[2],paddingShape[3]);

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
					dpadding.zero();
					backwardsBorder(batchIndex,channel, 0, 0, Ho, Wo);
					padding.backwardsChannel(dpadding, batchIndex,channel, gradientInput);
				}
			}
		} else {
			// Handle the case where there is at least one inner region which doesn't touch the border

			for (int batchIndex = 0; batchIndex < N; batchIndex++) {
				for( int channel = 0; channel < C; channel++ ) {
					dpadding.zero();

					// do the inner region first, which can be processed efficiently
					for (int outRow = outR0; outRow < outR1; outRow++) {
						int inputRow = outRow * config.periodY - paddingY0;

						for (int outCol = outC0; outCol < outC1; outCol++) {
							int inputCol = outCol * config.periodX - paddingX0;

							backwardsAt_inner(input, batchIndex, channel, inputRow, inputCol, outRow, outCol);
						}
					}
					// Process the borders, top, bottom, left, right
					backwardsBorder(batchIndex,channel, 0, 0, outR0, Wo);
					backwardsBorder(batchIndex,channel, outR1, 0, Ho, Wo);
					backwardsBorder(batchIndex,channel, outR0, 0, outR1, outC0);
					backwardsBorder(batchIndex,channel, outR0, outC1, outR1, Wo);

					// for this channel, go from gradient of padded image to gradient of input image
					padding.backwardsChannel(dpadding, batchIndex,channel, gradientInput);
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
	private void backwardsBorder( int batchIndex , int channel , int row0, int col0, int row1, int col1 ) {
		for (int outRow = row0; outRow < row1; outRow++) {
			int padRow = outRow*config.periodY;
			for (int outCol = col0; outCol < col1; outCol++) {
				int padCol = outCol*config.periodX;

				backwardsAt_border(padding, batchIndex, channel, padRow, padCol, outRow, outCol);
			}
		}
	}

	/**
	 * Applies the backwards local window operation.  The padded gradient (dpadding) should be computed here.
	 * The input gradient will be found later on
	 *
	 * @param input Input spatial tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param channel Channel
	 * @param inY y-axis lower extent, in input tensor coordinates
	 * @param inX x-axis lower extent, in input tensor coordinates
	 * @param outY y-axis output coordinate
	 * @param outX x-axis output coordinate
	 */
	protected abstract void backwardsAt_inner(T input, int batch, int channel,
											  int inY, int inX, int outY, int outX);

	/**
	 * Applies the backwards local window operation.  The padded gradient (dpadding) should be computed here.
	 * The input gradient will be found later on
	 *
	 * @param padded Input spatial virtual tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param channel Channel
	 * @param padY y-axis lower extent, in padded tensor coordinates
	 * @param padX x-axis lower extent, in padded tensor coordinates
	 * @param outY y-axis output coordinate
	 * @param outX x-axis output coordinate
	 */
	protected abstract void backwardsAt_border(P padded, int batch, int channel,
											   int padY, int padX, int outY, int outX);

	@Override
	public void learning() {
		learningMode = true;
	}

	@Override
	public void evaluating() {
		learningMode = false;
	}

	@Override
	public boolean isLearning() {
		return learningMode;
	}

	@Override
	public Class<T> getTensorType() {
		return padding.getTensorType();
	}
}
