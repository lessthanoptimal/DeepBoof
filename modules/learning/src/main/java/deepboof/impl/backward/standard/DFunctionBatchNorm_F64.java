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

import deepboof.backward.DFunctionBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DFunctionBatchNorm} for {@link Tensor_F64}.  Intermediate variables are cached in the
 * forward pass.
 *
 * @author Peter Abeles
 */
public class DFunctionBatchNorm_F64 extends BaseDBatchNorm_F64
        implements DFunctionBatchNorm<Tensor_F64>
{
	public DFunctionBatchNorm_F64(boolean requiresGammaBeta) {
		super(requiresGammaBeta);
	}

	@Override
	protected int[] createShapeVariables(int[] shapeInput) {
		return shapeInput;
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		if( input.length(0) <= 1 )
			throw new IllegalArgumentException("There must be more than 1 minibatch");

		if( learningMode ) {
			forwardsLearning(input, output);
		} else {
			forwardsEvaluate(input, output);
		}
	}

	private void forwardsLearning(Tensor_F64 input, Tensor_F64 output) {
		tensorDiffX.reshape( input.shape );
		tensorXhat.reshape( input.shape );
		computeStatisticsAndNormalize(input);

		if (requiresGammaBeta) {
			applyGammaBeta(output);
		} else {
			// is gamma and beta are not adjustable then the output is the normalized x_hat
			output.setTo(tensorXhat);
		}
	}

	public void forwardsEvaluate(Tensor_F64 input, Tensor_F64 output) {
		int D = TensorOps.outerLength(input.shape,1);

		int indexIn  = input.startIndex;
		int indexOut = output.startIndex;

		if( requiresGammaBeta ) {
			for (int batch = 0; batch < miniBatchSize; batch++) {
				int indexVar = 0;
				int indexP  = params.startIndex;
				int end = indexIn + D;
				while (indexIn < end) {
					double mean  = tensorMean.d[indexVar];
					double stdev_eps = tensorStd.d[indexVar];
					double gamma = params.d[indexP++];
					double beta  = params.d[indexP++];

					output.d[indexOut++] = (input.d[indexIn++] - mean)*(gamma / stdev_eps) + beta;
					indexVar++;
				}
			}
		} else {
			for (int stack = 0; stack < miniBatchSize; stack++) {
				int indexVar = 0;
				int end = indexIn + D;
				while (indexIn < end) {
					double mean  = tensorMean.d[indexVar];
					double stdev_eps = tensorStd.d[indexVar];

					output.d[indexOut++] = (input.d[indexIn++] - mean) / stdev_eps;
					indexVar++;
				}
			}
		}
	}

	/**
	 * Apply gamma and beta to normalized input x_hat
	 */
	private void applyGammaBeta(Tensor_F64 output) {
		int indexOut = output.startIndex;
		int indexTensor = 0;
		int end = params.length();

		for (int stack = 0; stack < miniBatchSize; stack++) {
			int indexParam = params.startIndex;
			while (indexParam < end) {
				double gamma = params.d[indexParam++];
				double beta = params.d[indexParam++];

				output.d[indexOut++] = gamma*tensorXhat.d[indexTensor++] + beta;
			}
		}
	}

	/**
	 * Computes and stores mean, standard deviation, and x_hat the normalized input vector
	 */
	private void computeStatisticsAndNormalize(Tensor_F64 input) {
		tensorMean.zero();
		tensorStd.zero();
		tensorXhat.zero();

		double M_var = miniBatchSize-1; // unbiased variance division, mean is computed with miniBatchSize

		// compute the mean
		int indexIn = input.startIndex;
		for (int stack = 0; stack < miniBatchSize; stack++) {
			int indexVar = 0;
			while (indexVar < D) {
				tensorMean.d[indexVar++] += input.d[indexIn++];
			}
		}
		for (int indexVar = 0; indexVar < D; indexVar++ ) {
			tensorMean.d[indexVar] /= miniBatchSize;
		}

		// compute the unbiased standard deviation with EPS for numerical reasons
		indexIn = input.startIndex;
		int indexTensor = 0;
		for (int stack = 0; stack < miniBatchSize; stack++) {
			for (int indexVar = 0; indexVar < D; indexVar++, indexTensor++ ) {
				double d = input.d[indexIn++] - tensorMean.d[indexVar];
				tensorDiffX.d[indexTensor] = d;
				tensorStd.d[indexVar] += d*d;
			}
		}
		for (int indexVar = 0; indexVar < D; indexVar++ ) {
			tensorStd.d[indexVar] = Math.sqrt( tensorStd.d[indexVar]/M_var + EPS);
		}

		// normalize so that mean is 1 and variance is 1
		// x_hat = (x - mu)/std
		indexTensor = 0;
		for (int stack = 0; stack < miniBatchSize; stack++) {
			for (int indexVar = 0; indexVar < D; indexVar++, indexTensor++ ) {
				tensorXhat.d[indexTensor] = tensorDiffX.d[indexTensor] / tensorStd.d[indexVar];
			}
		}
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
							  Tensor_F64 gradientInput,
							  List<Tensor_F64> gradientParameters)
	{
		// NOTE: @l/@y = dout
		tensorDXhat.reshape( input.shape );

		if( requiresGammaBeta ) {
			partialXHat(dout);
		} else {
			// if gamma and beta is not required then gamma effectively = 1 and Dxhat = dout
			tensorDXhat.setTo(dout);
		}

		partialVariance();
		partialMean();
		partialX(gradientInput);

		if( requiresGammaBeta ) {
			partialParameters(gradientParameters.get(0),dout);
		}
	}

	/**
	 * compute partial of gamma and Beta
	 *
	 * <pre> @l/@gamma = sum( @l/y[i]  * x_hat[i] ) </pre>
	 * <pre> @l/@Beta = sum( @l/y[i] )              </pre>
	 */
	private void partialParameters(Tensor_F64 tensorDParam , Tensor_F64 dout) {
		tensorDParam.zero();
		int indexDOut = dout.startIndex;
		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			int indexDParam = 0;
			for (int indexVar = 0; indexVar < D; indexVar++, indexTensor++, indexDOut++) {
				double d = dout.d[indexDOut];
				tensorDParam.d[indexDParam++] += d*tensorXhat.d[indexTensor];
				tensorDParam.d[indexDParam++] += d;
			}
		}
	}

	/**
	 * compute partial to x_hat
	 *
	 * <pre> @l/@x_hat[i] = @l/@y[i] * gamma  </pre>
	 */
	private void partialXHat(Tensor_F64 dout) {
		int indexDOut = dout.startIndex;
		for (int stack = 0,indexTensor = 0; stack < miniBatchSize; stack++) {
			for( int indexVar = 0; indexVar < D; indexVar++ , indexTensor++) {
				// see encoding of params
				tensorDXhat.d[indexTensor] = dout.d[indexDOut++]*params.d[indexVar*2];
			}
		}
	}

	/**
	 * compute partial of the input x
	 *
	 * <pre> @l/@x[i] = @l/@x_hat[i] / sqrt(sigma^2 + eps) + @l/@var * 2*(x[i]-mean)/M + @l/@mean * 1/M </pre>
	 */
	private void partialX( Tensor_F64 tensorDX ) {
		double M_var = miniBatchSize-1;
		int indexDX = tensorDX.startIndex;
		for (int stack = 0,indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int indexVar = 0; indexVar < D; indexVar++, indexTensor++, indexDX++ ) {
				double val = tensorDXhat.d[indexTensor] / tensorStd.d[indexVar];
				val += tensorDVar.d[indexVar]*2*tensorDiffX.d[indexTensor]/M_var + tensorDMean.d[indexVar]/miniBatchSize;

				tensorDX.d[indexDX] = val;
			}
		}
	}

	/**
	 * compute the mean partial
	 *
	 * <pre> @l/@mean = (sum( @l/@x_hat[i] * (-1/sqrt(var + EPS)) ) - @l/@var * (2/M) * sum( x[i] - mean )</pre>
	 */
	private void partialMean() {
		tensorDMean.zero();
		tensorTmp.zero();

		double M_var = miniBatchSize-1;

		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for( int indexVar = 0; indexVar < D; indexVar++, indexTensor++ ) {
				// sum( x[i] - mean )
				tensorTmp.d[indexVar] += tensorDiffX.d[indexTensor];
				// @l/@x[i] * (-1)
				tensorDMean.d[indexVar] -= tensorDXhat.d[indexTensor];
			}
		}

		for( int indexVar = 0; indexVar < D; indexVar++ ) {
			tensorDMean.d[indexVar] /= tensorStd.d[indexVar];
			tensorDMean.d[indexVar] -= 2.0*tensorDVar.d[indexVar]*tensorTmp.d[indexVar]/M_var;
		}
	}

	/**
	 * compute the variance partial
	 *
	 * <pre> @l/@var = sum( @l/@x_hat[i] * (x[i] - x_mean) *(-1/2)*(var + EPS)^(-3/2) </pre>
	 */
	private void partialVariance() {
		tensorDVar.zero();

		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for( int indexVar = 0; indexVar < D; indexVar++, indexTensor++ ) {
				// @l/@x_hat[i] * (x[i] - x_mean)
				tensorDVar.d[indexVar] += tensorDXhat.d[indexTensor]*tensorDiffX.d[indexTensor];
			}
		}

		// (-1/2)*(var + EPS)^(-3/2)
		for( int indexVar = 0; indexVar < D; indexVar++ ) {
			double sigmaPow3 = tensorStd.d[indexVar];
			sigmaPow3 = sigmaPow3*sigmaPow3*sigmaPow3;

			tensorDVar.d[indexVar] /= (-2.0*sigmaPow3);
		}

	}
}
