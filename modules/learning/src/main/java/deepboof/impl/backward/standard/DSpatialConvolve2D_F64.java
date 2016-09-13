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

import deepboof.backward.DSpatialConvolve2D;
import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.forward.ConfigConvolve2D;
import deepboof.tensors.Tensor_F64;

import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * Implementation of {@link DSpatialConvolve2D} for {@link Tensor_F64}.
 *
 * @author Peter Abeles
 */
public class DSpatialConvolve2D_F64
		extends DSpatialWindowImage<Tensor_F64,DSpatialPadding2D_F64>
		implements DSpatialConvolve2D<Tensor_F64>
{
	// see variable definitions in SpacialTensor2D javadoc
	protected int F; // number of kernels

	// Tensors extracted from parameters and output
	protected Tensor_F64 weights;
	protected Tensor_F64 bias;

	// cache used to store the local region in the input tensor which is being examined
	// reduces cache misses and can be used to store the image border
	protected double cacheLocal[] = new double[0];


	public DSpatialConvolve2D_F64(ConfigConvolve2D config, DSpatialPadding2D_F64 padding) {
		super(config, padding);

		this.F = config.F;
	}

	@Override
	public void _initialize() {
		super._initialize();

		shapeOutput = WI(F,Ho,Wo);

		// weights
		shapeParameters.add( WI(F,C,HH,WW) );
		// bias
		shapeParameters.add( WI(F) );
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		super.forwardImage(input, output);
	}

	@Override
	protected void forwardAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {
		cacheInner(input, batch, inY, inX);

		// perform convolution
		forwardCache(batch, outY, outX);
	}

	private void cacheInner(Tensor_F64 input, int batch, int inY, int inX) {
		int cacheIndex = 0;
		for (int channel = 0; channel < C; channel++) {
			int indexImageStart = input.idx(batch, channel, inY, inX);

			for (int kerY = 0; kerY < HH; kerY++) {
				int indexI = indexImageStart;

				for (int kerX = 0; kerX < WW; kerX++) {
					cacheLocal[cacheIndex++] = input.d[indexI++];
				}
				indexImageStart += W;
			}
		}
	}

	@Override
	protected void forwardAt_border(DSpatialPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
		// copy the local region into a cache
		cacheBorder(padded, batch, padY, padX);

		// perform convolution
		forwardCache(batch, outY, outX);
	}

	private void cacheBorder(DSpatialPadding2D_F64 padded, int batch, int padY, int padX) {
		int cacheIndex = 0;
		for (int channel = 0; channel < C; channel++) {
			for (int kerY = 0; kerY < HH; kerY++) {
				for (int kerX = 0; kerX < WW; kerX++) {
					cacheLocal[cacheIndex++] = padded.get(batch,channel, padY + kerY, padX + kerX);
				}
			}
		}
	}

	/**
	 * Convolves using the local spatial cache
	 */
	private void forwardCache(int batch, int outY, int outX) {
		final int length = C*HH*WW;
		final double d[] = weights.d; // appears to result in a very very small speed boost

		int indexW = weights.startIndex;

		for (int kernelIndex = 0; kernelIndex < F; kernelIndex++) {
			double sum = 0;
			int cacheIndex = 0;

			while( cacheIndex < length ) {
				sum += cacheLocal[cacheIndex++] * d[indexW++];
			}

			sum += bias.d[bias.idx(kernelIndex)];

			output.d[output.idx(batch, kernelIndex, outY, outX)] = sum;
		}
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

	}

	@Override
	protected void backwardsAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {
		cacheInner(input, batch, inY, inX);
	}

	@Override
	protected void backwardsAt_border(DSpatialPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
		cacheBorder(padded, batch, padY, padX);
	}

	private void backwardsCache(int batch, int outY, int outX) {
		final int length = C*HH*WW;
		final double d[] = weights.d; // appears to result in a very very small speed boost

		int indexW = weights.startIndex;

		for (int kernelIndex = 0; kernelIndex < F; kernelIndex++) {
			double sum = 0;
			int cacheIndex = 0;

			while( cacheIndex < length ) {
				sum += cacheLocal[cacheIndex++] * d[indexW++];
			}

			sum += bias.d[bias.idx(kernelIndex)];

			output.d[output.idx(batch, kernelIndex, outY, outX)] = sum;
		}
	}


	@Override
	public void _setParameters(List<Tensor_F64> parameters) {
		// input = (N,C,H,W), weights = (F, C, HH, WW), bias = (F,), output = (N, F, Hp, Wp)
		weights = parameters.get(0);
		bias = parameters.get(1);

		cacheLocal = new double[HH*WW*C];
	}

	@Override
	public ConfigConvolve2D getConfiguration() {
		return (ConfigConvolve2D)config;
	}
}
