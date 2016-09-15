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
 * <p>Local caches of the spatial tensor are used to reduce cache misses.  The cache will contain
 * a region across all of the tensor's channels that encompasses the region that a single convolution
 * would interact with.  In the backwards pass the local cache is written back into the derivative
 * padded input tensor.
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

	// gradient of parameters
	protected Tensor_F64 dWeights;
	protected Tensor_F64 dBias;

	// Reference to gradient from forward layer
	protected Tensor_F64 dout;

	// cache used to store the local region in the input tensor which is being examined
	// reduces cache misses and can be used to store the image border
	protected double cachedPadded[] = new double[0];
	protected double cachedDPadding[] = new double[0];


	public DSpatialConvolve2D_F64(ConfigConvolve2D config, DSpatialPadding2D_F64 padding) {
		super(config, padding);

		this.F = config.F;
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {
		// input = (N,C,H,W), weights = (F, C, HH, WW), bias = (F,), output = (N, F, Hp, Wp)
		weights = parameters.get(0);
		bias = parameters.get(1);

		cachedPadded = new double[HH*WW*C];
		cachedDPadding = new double[HH*WW*C];
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
		tensorToCache(input, batch, inY, inX,cachedPadded);

		// perform convolution
		forwardCache(batch, outY, outX);
	}

	@Override
	protected void forwardAt_border(DSpatialPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
		// copy the local region into a cache
		borderToCache(padded, batch, padY, padX);

		// perform convolution
		forwardCache(batch, outY, outX);
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
				sum += cachedPadded[cacheIndex++] * d[indexW++];
			}

			sum += bias.d[bias.idx(kernelIndex)];

			output.d[output.idx(batch, kernelIndex, outY, outX)] = sum;
		}
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

		dWeights = gradientParameters.get(0);
		dBias = gradientParameters.get(1);

		dWeights.zero();
		dBias.zero();

		this.dout = dout;

		backwardsImage(input, gradientInput);
	}

	@Override
	protected void backwardsAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {
		tensorToCache(input, batch, inY, inX, cachedPadded);

		// compute pixel location in padded image
		int padY = outY*config.periodY;
		int padX = outX*config.periodX;

		// dpadding is a 3-dof fredom and doesn't have mini-batch
		tensorToCache(dpadding,-1,padY, padX, cachedDPadding);

		// perform calculation on cached tensors
		backwardsCache(batch,outY,outX);

		// save changes to dpadding
		cacheToTensor(cachedDPadding,dpadding,padY, padX);
	}

	@Override
	protected void backwardsAt_border(DSpatialPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
		borderToCache(padded, batch, padY, padX);
		tensorToCache(dpadding,-1, padY, padX, cachedDPadding);
		backwardsCache(batch,outY,outX);
		cacheToTensor(cachedDPadding,dpadding,padY, padX);
	}

	/**
	 * Perform backwards for a single convolution of each kernel using the locally cached padded
	 * tensor and it's cached derivative
	 */
	private void backwardsCache(int batch, int outY, int outX) {
		final int length = C*HH*WW;
		final double d[] = weights.d; // appears to result in a very very small speed boost

		int indexW = weights.startIndex;
		int indexD = bias.startIndex;
		int dweightsIndex = dWeights.startIndex;

		for (int kernelIndex = 0; kernelIndex < F; kernelIndex++) {
			int cacheIndex = 0;
			double val_dout = dout.d[dout.idx(batch,kernelIndex,outY,outX)];

			while( cacheIndex < length ) {
				double x = cachedPadded[cacheIndex];
				double w = d[indexW++];
				cachedDPadding[cacheIndex] += w*val_dout;
				dWeights.d[dweightsIndex++] += x*val_dout;

				cacheIndex++;
			}

			dBias.d[indexD++] += val_dout;
		}
	}

	private void tensorToCache(Tensor_F64 input, int batch, int inY, int inX,
							   double cache[] ) {
		int cacheIndex = 0;
		int stride = input.length(-1);
		for (int channel = 0; channel < C; channel++) {
			int indexImageStart = batch >= 0 ? input.idx(batch, channel, inY, inX)
					: input.idx(channel, inY, inX);

			for (int kerY = 0; kerY < HH; kerY++) {
				int indexI = indexImageStart;

				for (int kerX = 0; kerX < WW; kerX++) {
					cache[cacheIndex++] = input.d[indexI++];
				}
				indexImageStart += stride;
			}
		}
	}

	private void cacheToTensor(double cache[],
							   Tensor_F64 input, int inY, int inX ) {
		int cacheIndex = 0;
		int stride = input.length(-1);
		for (int channel = 0; channel < C; channel++) {
			int indexImageStart = input.idx(channel, inY, inX);

			for (int kerY = 0; kerY < HH; kerY++) {
				int indexI = indexImageStart;

				for (int kerX = 0; kerX < WW; kerX++) {
					input.d[indexI++] = cache[cacheIndex++];
				}
				indexImageStart += stride;
			}
		}
	}

	private void borderToCache(DSpatialPadding2D_F64 padded, int batch, int padY, int padX) {
		int cacheIndex = 0;
		for (int channel = 0; channel < C; channel++) {
			for (int kerY = 0; kerY < HH; kerY++) {
				for (int kerX = 0; kerX < WW; kerX++) {
					cachedPadded[cacheIndex++] = padded.get(batch,channel, padY + kerY, padX + kerX);
				}
			}
		}
	}

	@Override
	public ConfigConvolve2D getConfiguration() {
		return (ConfigConvolve2D)config;
	}
}
