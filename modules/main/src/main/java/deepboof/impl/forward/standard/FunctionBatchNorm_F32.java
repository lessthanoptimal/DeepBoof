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

import deepboof.DeepBoofConstants;
import deepboof.forward.FunctionBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F32;

import java.util.List;

/**
 * Implementation of {@link FunctionBatchNorm} for {@link Tensor_F32}.
 *
 * @author Peter Abeles
 */
public class FunctionBatchNorm_F32
		extends BaseFunction<Tensor_F32>
		implements FunctionBatchNorm<Tensor_F32>
{
	protected boolean requiresGammaBeta;

	// internal copy of parameters with variance modified for performance.  precomputes inverse of stdev + EPS
	protected Tensor_F32 params = new Tensor_F32(0);
	protected float EPS = DeepBoofConstants.TEST_TOL_F32*0.1f;

	public FunctionBatchNorm_F32(boolean requiresGammaBeta) {
		this.requiresGammaBeta = requiresGammaBeta;
	}

	@Override
	public void _initialize() {
		this.shapeOutput = shapeInput.clone();

		int shapeParam[] = TensorOps.WI( shapeInput, requiresGammaBeta ? 4 : 2 );

		this.shapeParameters.add(shapeParam);
		params.reshape(shapeParam);
	}

	@Override
	public void _setParameters(List<Tensor_F32> parameters) {
		params.setTo(parameters.get(0));
		int N = params.length();
		int stride = requiresGammaBeta ? 4 : 2;

		for (int i = 1; i < N; i += stride) {
			params.d[i] = 1.0f / (float)Math.sqrt(params.d[i] + EPS);
		}
	}

	@Override
	public void _forward(Tensor_F32 input, Tensor_F32 output) {
		if( input.getDimension() <= 1 ) {
			throw new IllegalArgumentException("Input tensor must be at least 2D.  First dimension of batch.");
		}

		int D = TensorOps.outerLength(input.shape,1);

		int indexIn  = input.startIndex;
		int indexOut = output.startIndex;

		if( requiresGammaBeta ) {
			for (int batch = 0; batch < miniBatchSize; batch++) {
				int indexP  = params.startIndex;
				int end = indexIn + D;
				while (indexIn < end) {
					float mean  = params.d[indexP++];
					float inv_stdev_eps = params.d[indexP++];
					float gamma = params.d[indexP++];
					float beta  = params.d[indexP++];

					output.d[indexOut++] = (input.d[indexIn++] - mean)*(gamma * inv_stdev_eps) + beta;
				}
			}
		} else {
			for (int stack = 0; stack < miniBatchSize; stack++) {
				int indexP  = params.startIndex;
				int end = indexIn + D;
				while (indexIn < end) {
					float mean = params.d[indexP++];
					float inv_stdev_eps = params.d[indexP++];

					output.d[indexOut++] = (input.d[indexIn++] - mean) * inv_stdev_eps;
				}
			}
		}
	}

	@Override
	public /**/double getEPS() {
		return EPS;
	}

	@Override
	public void setEPS( /**/double EPS) {
		this.EPS = (float)EPS;
	}

	@Override
	public boolean hasGammaBeta() {
		return requiresGammaBeta;
	}

	@Override
	public Class<Tensor_F32> getTensorType() {
		return Tensor_F32.class;
	}
}
