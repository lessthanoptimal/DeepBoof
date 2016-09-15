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

import deepboof.backward.DSpatialBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DSpatialBatchNorm} for {@link Tensor_F64}.
 *
 * @author Peter Abeles
 */
public class DSpatialBatchNorm_F64 extends BaseDBatchNorm_F64
		implements DSpatialBatchNorm<Tensor_F64>
{
	int numChannels;
	int numPixels;

	// Number of elements statistics are computed from
	// M_var is M-1 for computing unbiased variance
	double M,M_var;

	public DSpatialBatchNorm_F64(boolean requiresGammaBeta) {
		super(requiresGammaBeta);
	}

	@Override
	protected int[] createShapeVariables(int[] shapeInput) {
		return new int[]{shapeInput[0]}; // one variable for each channel
	}

	// TODO push into base class?
	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		if( input.length(0) <= 1 )
			throw new IllegalArgumentException("There must be more than 1 minibatch");

		tensorDiffX.reshape( input.shape );
		tensorXhat.reshape( input.shape );

		// just compute these variables onces.  They are used all over the place
		numChannels = input.length(1);
		numPixels = TensorOps.outerLength(input.shape,2);
		M = miniBatchSize*numPixels;
		M_var = M-1;

		if( learningMode ) {
			forwardLearning(input, output);
		} else {
			forwardEvaluate(input, output);
		}
	}

	private void forwardLearning(Tensor_F64 input, Tensor_F64 output) {
		computeStatisticsAndNormalize(input);

		if (requiresGammaBeta) {
			applyGammaBeta(output);
		} else {
			// is gamma and beta are not adjustable then the output is the normalized x_hat
			output.setTo(tensorXhat);
		}
	}

	public void forwardEvaluate(Tensor_F64 input, Tensor_F64 output) {
		int C = input.length(1);
		int W = input.length(2);
		int H = input.length(3);

		int D = W*H;

		int indexIn  = input.startIndex;
		int indexOut = output.startIndex;

		if( hasGammaBeta() ) {
			for (int batch = 0; batch < miniBatchSize; batch++) {
				int indexP  = params.startIndex;
				for( int channel = 0; channel < C; channel++ ) {
					double mean  = tensorMean.d[channel];
					double stdev_eps = tensorStd.d[channel];
					double gamma = params.d[indexP++];
					double beta  = params.d[indexP++];

					int end = indexIn + D;
					while (indexIn < end) {
						output.d[indexOut++] = (input.d[indexIn++] - mean)*(gamma / stdev_eps) + beta;
					}
				}
			}
		} else {
			for (int batch = 0; batch < miniBatchSize; batch++) {
				for (int channel = 0; channel < C; channel++) {
					double mean  = tensorMean.d[channel];
					double stdev_eps = tensorStd.d[channel];

					int end = indexIn + D;
					while (indexIn < end) {
						output.d[indexOut++] = (input.d[indexIn++] - mean) / stdev_eps;
					}
				}
			}
		}
	}

	/**
	 * Apply gamma and beta to normalized input x_hat
	 */
	private void applyGammaBeta(Tensor_F64 output) {

		int indexOut = output.startIndex;

		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double gamma = params.d[channel*2];
				double beta = params.d[channel*2+1];

				for (int pixel = 0; pixel < numPixels; pixel++) {
					output.d[indexOut++] = gamma*tensorXhat.d[indexTensor++] + beta;
				}
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

		// compute the mean
		int indexIn = input.startIndex;
		for (int stack = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double sum = 0;
				for (int pixel = 0; pixel < numPixels; pixel++) {
					sum += input.d[indexIn++];
				}
				tensorMean.d[channel] += sum;
			}
		}
		for (int channel = 0; channel < numChannels; channel++) {
			tensorMean.d[channel] /= M;
		}

		// compute the unbiased standard deviation with EPS for numerical reasons
		indexIn = input.startIndex;
		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double sum = 0;
				double channelMean = tensorMean.d[channel];
				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++ ) {
					double d = input.d[indexIn++] - channelMean;
					tensorDiffX.d[indexTensor] = d;
					sum += d*d;
				}
				tensorStd.d[channel] += sum;
			}
		}
		for (int channel = 0; channel < numChannels; channel++) {
			tensorStd.d[channel] = Math.sqrt( tensorStd.d[channel]/M_var + EPS);
		}

		// normalize so that mean is 1 and variance is 1
		// x_hat = (x - mu)/std
		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double channelStd = tensorStd.d[channel];

				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++ ) {
					tensorXhat.d[indexTensor] = tensorDiffX.d[indexTensor] / channelStd;
				}
			}
		}
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

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
			for (int channel = 0; channel < numChannels; channel++) {
				double sumDGamma = 0;
				double sumDBeta = 0;

				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++, indexDOut++) {
					double d = dout.d[indexDOut];
					sumDGamma += d*tensorXhat.d[indexTensor];
					sumDBeta += d;
				}

				tensorDParam.d[indexDParam++] += sumDGamma;
				tensorDParam.d[indexDParam++] += sumDBeta;
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
        
		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double gamma = params.d[channel*2];
				for (int pixel = 0; pixel < numPixels; pixel++) {
					tensorDXhat.d[indexTensor++] = dout.d[indexDOut++]*gamma;
				}
			}
		}
	}

	/**
	 * compute partial of the input x
	 *
	 * <pre> @l/@x[i] = @l/@x_hat[i] / sqrt(sigma^2 + eps) + @l/@var * 2*(x[i]-mean)/M + @l/@mean * 1/M </pre>
	 */
	private void partialX( Tensor_F64 tensorDX ) {
        
		int indexDX = tensorDX.startIndex;
		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double stdev = tensorStd.d[channel];
				double dvar = tensorDVar.d[channel];
				double dmean = tensorDMean.d[channel];

				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++, indexDX++) {
					double val = tensorDXhat.d[indexTensor]/stdev;
					val += dvar*2*tensorDiffX.d[indexTensor]/M_var + dmean/M;

					tensorDX.d[indexDX] = val;
				}
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

		for (int stack = 0, indexTensor = 0; stack < miniBatchSize; stack++) {
			for (int channel = 0; channel < numChannels; channel++) {
				double sumTmp = 0;
				double sumDMean = 0;
				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++) {
					// sum( x[i] - mean )
					sumTmp += tensorDiffX.d[indexTensor];
					// @l/@x[i] * (-1)
					sumDMean -= tensorDXhat.d[indexTensor];
				}

				tensorTmp.d[channel] += sumTmp;
				tensorDMean.d[channel] += sumDMean;
			}
		}

		for (int channel = 0; channel < numChannels; channel++) {
			tensorDMean.d[channel] /= tensorStd.d[channel];
			tensorDMean.d[channel] -= 2.0*tensorDVar.d[channel]*tensorTmp.d[channel]/M_var;
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
			for (int channel = 0; channel < numChannels; channel++) {
				double sumDVar = 0;
				for (int pixel = 0; pixel < numPixels; pixel++, indexTensor++) {
					// @l/@x_hat[i] * (x[i] - x_mean)
					sumDVar += tensorDXhat.d[indexTensor]*tensorDiffX.d[indexTensor];
				}
				tensorDVar.d[channel] += sumDVar;
			}
		}

		// (-1/2)*(var + EPS)^(-3/2)
		for (int channel = 0; channel < numChannels; channel++) {
			double sigmaPow3 = tensorStd.d[channel];
			sigmaPow3 = sigmaPow3*sigmaPow3*sigmaPow3;

			tensorDVar.d[channel] /= (-2.0*sigmaPow3);
		}

	}
}
