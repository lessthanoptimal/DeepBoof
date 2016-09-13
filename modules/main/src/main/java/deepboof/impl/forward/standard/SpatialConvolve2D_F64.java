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

import deepboof.forward.ConfigConvolve2D;
import deepboof.forward.SpatialConvolve2D;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * Implementation of {@link SpatialConvolve2D} for {@link Tensor_F64}
 *
 * @author Peter Abeles
 */
public class SpatialConvolve2D_F64
		extends SpatialWindowImage<Tensor_F64,SpatialPadding2D_F64>
		implements SpatialConvolve2D<Tensor_F64>
{

	// see variable definitions in SpacialTensor2D javadoc
	protected int F; // number of kernels

	// Tensors extracted from parameters and output
	protected Tensor_F64 weights;
	protected Tensor_F64 bias;

	// cache used to store the local region in the input tensor which is being examined
	// reduces cache misses and can be used to store the image border
	protected double cacheLocal[] = new double[0];

	public SpatialConvolve2D_F64(ConfigConvolve2D config,
								 SpatialPadding2D_F64 padding ) {
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
	public void _setParameters(List<Tensor_F64> parameters) {
		// input = (N,C,H,W), weights = (F, C, HH, WW), bias = (F,), output = (N, F, Hp, Wp)
		weights = parameters.get(0);
		bias = parameters.get(1);

		cacheLocal = new double[HH*WW*C];
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		super.forwardImage(input, output);
	}

	@Override
	protected void forwardAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {
		// copy the local region into a cache
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

		// perform convolution
		convolveCache(batch, outY, outX);
	}

	@Override
	protected void forwardAt_border(SpatialPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
		// copy the local region into a cache
		int cacheIndex = 0;
		for (int channel = 0; channel < C; channel++) {
			for (int kerY = 0; kerY < HH; kerY++) {
				for (int kerX = 0; kerX < WW; kerX++) {
					cacheLocal[cacheIndex++] = padded.get(batch,channel, padY + kerY, padX + kerX);
				}
			}
		}

		// perform convolution
		convolveCache(batch, outY, outX);
	}

	/**
	 * Convolves using the local spatial cache
	 */
	private void convolveCache(int batch, int outY, int outX) {
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
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}

	@Override
	public ConfigConvolve2D getConfiguration() {
		return (ConfigConvolve2D)config;
	}
}
