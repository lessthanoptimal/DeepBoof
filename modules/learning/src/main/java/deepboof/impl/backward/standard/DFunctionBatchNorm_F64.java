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

        tensorDiffX.reshape( input.shape );
        tensorXhat.reshape( input.shape );

        computeStatisticsAndNormalize(input);

        if( requiresGammaBeta ) {
            applyGammaBeta(output);
        } else {
            // is gamma and beta are not adjustable then the output is the normalized x_hat
            output.setTo(tensorXhat);
        }
    }

    /**
     * Apply gamma and beta to normalized input x_hat
     */
    private void applyGammaBeta(Tensor_F64 output) {
        int indexOut = output.startIndex;
        int indexXHat = 0;
        int end = params.length();

        for (int stack = 0; stack < miniBatchSize; stack++) {
            int indexParam = params.startIndex;
            while (indexParam < end) {
                double gamma = params.d[indexParam++];
                double beta = params.d[indexParam++];

                output.d[indexOut++] = gamma*tensorXhat.d[indexXHat++] + beta;
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
            int indexMV = 0;
            while (indexMV < D) {
                tensorMean.d[indexMV++] += input.d[indexIn++];
            }
        }
        for (int indexMV = 0; indexMV < D; indexMV++ ) {
            tensorMean.d[indexMV] /= miniBatchSize;
        }

        // compute the unbiased standard deviation with EPS for numerical reasons
        indexIn = input.startIndex;
        int indexDiffX = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for (int indexMV = 0; indexMV < D; indexMV++, indexDiffX++ ) {
                double d = input.d[indexIn++] - tensorMean.d[indexMV];
                tensorDiffX.d[indexDiffX] = d;
                tensorStd.d[indexMV] += d*d;
            }
        }
        for (int indexMV = 0; indexMV < D; indexMV++ ) {
            tensorStd.d[indexMV] = Math.sqrt( tensorStd.d[indexMV]/M_var + EPS);
        }

        // normalize so that mean is 1 and variance is 1
        // x_hat = (x - mu)/std
        indexDiffX = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for (int indexMV = 0; indexMV < D; indexMV++, indexDiffX++ ) {
                tensorXhat.d[indexDiffX] = tensorDiffX.d[indexDiffX] / tensorStd.d[indexMV];
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
        int indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            int indexDParam = 0;
            for (int indexMV = 0; indexMV < D; indexMV++, indexXHat++, indexDOut++) {
                double d = dout.d[indexDOut];
                tensorDParam.d[indexDParam++] += d*tensorXhat.d[indexXHat];
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
        int indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for( int indexMV = 0; indexMV < D; indexMV++ ) {
                // see encoding of params
                tensorDXhat.d[indexXHat++] = dout.d[indexDOut++]*params.d[indexMV*2];
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
        int indexXHat = 0;
        int indexX = tensorDX.startIndex;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for (int indexMV = 0; indexMV < D; indexMV++, indexXHat++, indexX++ ) {
                double val = tensorDXhat.d[indexXHat] / tensorStd.d[indexMV];
                val += tensorDVar.d[indexMV]*2*tensorDiffX.d[indexXHat]/M_var + tensorDMean.d[indexMV]/miniBatchSize;

                tensorDX.d[indexX] = val;
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

        int indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for( int indexMV = 0; indexMV < D; indexMV++, indexXHat++ ) {
                // sum( x[i] - mean )
                tensorTmp.d[indexMV] += tensorDiffX.d[indexXHat];
                // @l/@x[i] * (-1)
                tensorDMean.d[indexMV] -= tensorDXhat.d[indexXHat];
            }
        }

        for( int indexMV = 0; indexMV < D; indexMV++ ) {
            tensorDMean.d[indexMV] /= tensorStd.d[indexMV];
            tensorDMean.d[indexMV] -= 2.0*tensorDVar.d[indexMV]*tensorTmp.d[indexMV]/M_var;
        }
    }

    /**
     * compute the variance partial
     *
     * <pre> @l/@var = sum( @l/@x_hat[i] * (x[i] - x_mean) *(-1/2)*(var + EPS)^(3/2) </pre>
     */
    private void partialVariance() {
        tensorDVar.zero();

        int indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for( int indexMV = 0; indexMV < D; indexMV++, indexXHat++ ) {
                double x_m_mean = tensorDiffX.d[indexXHat];
                tensorDVar.d[indexMV] += tensorDXhat.d[indexXHat]*x_m_mean;
            }
        }

        for( int indexMV = 0; indexMV < D; indexMV++, indexXHat++ ) {
            double sigmaPow3 = tensorStd.d[indexMV];
            sigmaPow3 = sigmaPow3*sigmaPow3*sigmaPow3;

            tensorDVar.d[indexMV] /= (-2.0*sigmaPow3);
        }

    }
}
