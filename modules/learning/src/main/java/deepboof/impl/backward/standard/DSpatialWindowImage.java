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
import deepboof.impl.forward.standard.SpatialWindowImage;
import deepboof.misc.TensorFactory;
import deepboof.misc.TensorOps;

import java.util.List;

/**
 * Backwards functions for operations which convolve a window across the input spatial tensor.
 * Each image in a mini batch is processed one at a time.
 *
 * @author Peter Abeles
 */
public abstract class DSpatialWindowImage<T extends Tensor<T>, P extends DSpatialPadding2D<T>>
		extends SpatialWindowImage<T,P> implements DFunction<T> {

	// Toggle indicating if it's in learning mode or not
	protected boolean learningMode = false;

	// storage for padded image gradient.  This is a 2D tensor
	protected T dpadding;

	public DSpatialWindowImage(ConfigSpatial config, P padding) {
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

	protected void backwardsImage(T input, T gradientInput) {
		padding.setInput(input);

		// only need to do the spatial component for 1 channel
		int[] paddingShape = padding.getShape();
		dpadding.reshape(paddingShape[1],paddingShape[2],paddingShape[3]);

		// extract constants which describe the convolution from inputs and parameters
		N = input.length(0);

		int paddingX0 = padding.getPaddingCol0();
		int paddingY0 = padding.getPaddingRow0();

		// lower and upper extends for where the input image is inside of the padded image
		int outC0 = innerLowerExtent(config.periodX,paddingX0);
		int outC1 = innerUpperExtent(config.WW,config.periodX,paddingX0,W);
		int outR0 = innerLowerExtent(config.periodY,paddingY0);
		int outR1 = innerUpperExtent(config.HH,config.periodY,paddingY0,H);

		if( isEntirelyBorder(outR0, outC0) ) {
			// Handle the case where the entire output touches the border
			for (int batchIndex = 0; batchIndex < N; batchIndex++) {
				dpadding.zero();
				backwardsBorder(batchIndex, 0, 0, Ho, Wo);
				padding.backwardsImage(dpadding, batchIndex,gradientInput);
			}
		} else {
			// Handle the case where there is at least one inner region which doesn't touch the border

			for (int batchIndex = 0; batchIndex < N; batchIndex++) {
				dpadding.zero();

				// do the inner region first, which can be processed efficiently
				for (int outRow = outR0; outRow < outR1; outRow++) {
					int inputRow = outRow * config.periodY - paddingY0;

					for (int outCol = outC0; outCol < outC1; outCol++) {
						int inputCol = outCol * config.periodX - paddingX0;

						backwardsAt_inner(input, batchIndex, inputRow, inputCol, outRow, outCol);
					}
				}

				// Process the borders, top, bottom, left, right
				backwardsBorder(batchIndex, 0, 0, outR0, Wo);
				backwardsBorder(batchIndex, outR1, 0, Ho, Wo);
				backwardsBorder(batchIndex, outR0, 0, outR1, outC0);
				backwardsBorder(batchIndex, outR0, outC1, outR1, Wo);

				padding.backwardsImage(dpadding, batchIndex,gradientInput);
			}
		}
	}

	/**
	 * Processes along the spatial border border.
	 *
	 * @param batchIndex which mini-batch
	 * @param row0 Lower extent along rows, inclusive
	 * @param col0 Lower extent along columns, inclusive
	 * @param row1 Upper extent along rows, exclusive
	 * @param col1 Upper extent along columns, exclusive
	 */
	private void backwardsBorder(int batchIndex , int row0, int col0, int row1, int col1 ) {
		for (int outRow = row0; outRow < row1; outRow++) {
			int paddedRow = outRow*config.periodY;
			for (int outCol = col0; outCol < col1; outCol++) {
				int paddedCol = outCol*config.periodX;

				backwardsAt_border(padding, batchIndex, paddedRow, paddedCol, outRow, outCol);
			}
		}
	}

	/**
	 * Applies the operations at the specified window and stores the results at the specified output
	 * coordinate.
	 *
	 * @param input Input spatial tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param inY y-axis lower extent, in input coordinates
	 * @param inX x-axis lower extent, in input coordinates
	 * @param outY y-axis output coordinates
	 * @param outX x-axis output coordinates
	 */
	protected abstract void backwardsAt_inner(T input, int batch, int inY, int inX, int outY, int outX);

	/**
	 * Applies the operations at the specified window and stores the results at the specified output
	 * coordinate.  For virtual tensor
	 *
	 * @param padded Padded input spatial virtual tensor
	 * @param batch Index of input in mini-batch that is being processed
	 * @param padY y-axis lower extent, inclusive.  Padded coordinates
	 * @param padX x-axis lower extent, inclusive.  Padded coordinates
	 * @param outY y-axis output coordinates
	 * @param outX x-axis output coordinates
	 */
	protected abstract void backwardsAt_border(P padded, int batch, int padY, int padX, int outY, int outX);


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
