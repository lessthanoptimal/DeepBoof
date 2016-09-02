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

import deepboof.DeepBoofConstants;
import deepboof.backward.DFunctionBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DFunctionBatchNorm} for {@link Tensor_F64}.  In the forward pass BLAH is cached
 * for use in the backwards pass.
 *
 * @author Peter Abeles
 */
public class DFunctionBatchNorm_F64 extends BaseDFunction<Tensor_F64> implements DFunctionBatchNorm<Tensor_F64>
{
    // If gamma and beta are used as parameters
    protected boolean requiresGammaBeta;

    // storage for mean and standard deviation tensor
    protected Tensor_F64 tensorMean = new Tensor_F64();
    protected Tensor_F64 tensorStd = new Tensor_F64();  // really is sqrt( stdev^2 + eps) ~= stdev
    // storage for the normalized input  (e.g. stdev = 1, mean = 1)
    protected Tensor_F64 tensorXhat = new Tensor_F64();

    // storage for gradient of variance, mean, and others
    protected Tensor_F64 tensorDVar = new Tensor_F64();
    protected Tensor_F64 tensorDMean = new Tensor_F64();
    protected Tensor_F64 tensorDXhat = new Tensor_F64();
    protected Tensor_F64 tensorDParam = new Tensor_F64();

    // x[i] - mean(x)
    protected Tensor_F64 tensorDiffX = new Tensor_F64();

    // temporary storage
    protected Tensor_F64 tensorTmp = new Tensor_F64();

    // number of elements in input tensor (excluding mini-batch)
    // NOTE: In the paper this variable is 'm'
    private int D;

    // Internal storage for gamma and beta parameters.  Stored interleaved gamma then beta.  1 for each input variable
    // params = [ gamma[0], beta[0], gamma[1], beta[1],  ... , gamma[D], beta[D]]
    protected Tensor_F64 params = new Tensor_F64(0);
    protected double EPS = DeepBoofConstants.TEST_TOL_F64*0.1;

    public DFunctionBatchNorm_F64(boolean requiresGammaBeta) {
        this.requiresGammaBeta = requiresGammaBeta;
    }

    @Override
    public void _initialize() {
        tensorMean.reshape( shapeInput );
        tensorStd.reshape( shapeInput );
        tensorDVar.reshape( shapeInput );
        tensorDMean.reshape( shapeInput );
        tensorTmp.reshape( shapeInput );

        this.shapeOutput = shapeInput.clone();

        if( requiresGammaBeta ) {
            int shapeParam[] = TensorOps.WI(shapeInput, 2);
            this.shapeParameters.add(shapeParam);
            params.reshape(shapeParam);
            tensorDParam.reshape(shapeParam);
        }

        D = TensorOps.tensorLength(shapeInput);
    }

    @Override
    public void _setParameters(List<Tensor_F64> parameters) {
        if( requiresGammaBeta ) {
            params.setTo(parameters.get(0));
        } else if( parameters.size() != 0 ){
            throw new IllegalArgumentException("There are no parameters since gamma and beta have been turned off");
        }
    }

    @Override
    public void _forward(Tensor_F64 input, Tensor_F64 output) {
        if( input.length(0) <= 1 )
            throw new IllegalArgumentException("There must be more than 1 minibatch");

        tensorDiffX.reshape( input.shape );
        tensorDXhat.reshape( input.shape );

        if( requiresGammaBeta ) {
            tensorXhat.reshape( input.shape );
            computeStatisticsAndNormalize(input,tensorXhat);
            applyGammaBeta(output);
        } else {
            // is gamma and beta is off then the output is the normalized x_hat
            computeStatisticsAndNormalize(input,output);
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
    private void computeStatisticsAndNormalize(Tensor_F64 input, Tensor_F64 tensorXhat) {
        tensorMean.zero();
        tensorStd.zero();
        tensorXhat.zero();

        // length of the tensor outside of the batch norm
        int D = TensorOps.outerLength(input.shape,1);

        // compute the mean
        int indexIn = input.startIndex;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            int indexMV = 0;
            while (indexMV < D) {
                tensorMean.d[indexMV++] += input.d[indexIn++];
            }
        }
        int indexMV = 0;
        while (indexMV < D) {
            tensorMean.d[indexMV++] /= miniBatchSize;
        }

        // compute the unbiased standard deviation with EPS for numerical reasons
        indexIn = input.startIndex;
        int indexDiffX = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            indexMV = 0;
            while (indexMV < D) {
                double d = input.d[indexIn++] - tensorMean.d[indexMV];
                tensorDiffX.d[indexDiffX++] = d;
                tensorStd.d[indexMV++] += d*d;
            }
        }
        indexMV = 0;
        while (indexMV < D) {
            tensorStd.d[indexMV] = Math.sqrt( tensorStd.d[indexMV]/(miniBatchSize-1) + EPS);
            indexMV++;
        }

        // normalize so that mean is 1 and variance is 1
        // x_hat = (x - mu)/std
        indexIn = input.startIndex;
        int indexXHat = tensorXhat.startIndex;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            indexMV = 0;
            while (indexMV < D) {
                tensorXhat.d[indexXHat++] = (input.d[indexIn++] - tensorMean.d[indexMV]) / tensorStd.d[indexMV];
                indexMV++;
            }
        }
    }

    @Override
    protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

        // NOTE: @l/@y = dout

        Tensor_F64 tensorDXhat;

        if( requiresGammaBeta ) {
            partialXHat(dout);
            tensorDXhat = this.tensorDXhat;
        } else {
            // if gamma and beta is not required then gamma effectively = 1 and Dxhat = dout
            tensorDXhat = dout;
        }

        partialVariance(tensorDXhat);
        partialMean(tensorDXhat);
        partialX(gradientInput, tensorDXhat);

        if( requiresGammaBeta ) {
            partialParameters(dout);
        }
    }

    /**
     * compute partial of gamma and Beta
     *
     * <pre> @l/@gamma = sum( @l/y[i]  * x_hat[i] ) </pre>
     * <pre> @l/@Beta = sum( @l/y[i] )              </pre>
     */
    private void partialParameters(Tensor_F64 dout) {
        tensorDParam.zero();
        int indexDOut = dout.startIndex;
        int indexXHat = 0;
        int indexDParam = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
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
     * <pre> @l/@x[i] = @l/@x_hat[i] / sqrt(sigma^2 + eps) + @l/@var * 2*(x[i]-mean)/D + @l/@mean * 1/D </pre>
     */
    private void partialX( Tensor_F64 tensorDX , Tensor_F64 tensorDXhat ) {
        int indexXHat = 0;
        int indexX = tensorDX.startIndex;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for (int indexMV = 0; indexMV < D; indexMV++, indexXHat++, indexX++ ) {
                double val = tensorDXhat.d[indexXHat] / tensorStd.d[indexMV];
                val += tensorDVar.d[indexMV]*2*tensorDiffX.d[indexXHat]/D + tensorDMean.d[indexMV]/D;

                tensorDX.d[indexX] = val;
            }
        }
    }

    /**
     * compute the mean partial
     *
     * <pre> @l/@mean = (sum( @l/@x_hat[i] * (-1/sqrt(var + EPS)) ) - @l/@var * (2/D) * sum( x[i] - mean )</pre>
     */
    private void partialMean( Tensor_F64 tensorDXhat  ) {
        tensorDMean.zero();
        tensorTmp.zero();

        int indexXHat;
        indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for( int indexMV = 0; indexMV < D; indexMV++, indexXHat++ ) {
                // sum( x[i] - mean )
                tensorTmp.d[indexMV] += tensorDiffX.d[indexXHat];
                // @l/@x[i] * (-1/sqrt(var + EPS))
                tensorDMean.d[indexMV] -= tensorDXhat.d[indexXHat]/tensorStd.d[indexMV];
            }
        }

        for( int indexMV = 0; indexMV < D; indexMV++ ) {
            tensorDMean.d[indexMV] -= 2.0*tensorDVar.d[indexMV]*tensorTmp.d[indexMV]/D;
        }
    }

    /**
     * compute the variance partial
     *
     * <pre> @l/@var = sum( @l/@x_hat[i] * (x[i] - x_mean) *(-1/2)*(var + EPS)^(3/2) </pre>
     */
    private void partialVariance( Tensor_F64 tensorDXhat ) {
        tensorDVar.zero();

        int indexXHat;
        indexXHat = 0;
        for (int stack = 0; stack < miniBatchSize; stack++) {
            for( int indexMV = 0; indexMV < D; indexMV++, indexXHat++ ) {
                double x_m_mean = tensorDiffX.d[indexXHat];
                double sigmaPow3 = tensorStd.d[indexMV];
                sigmaPow3 = sigmaPow3*sigmaPow3*sigmaPow3;
                tensorDVar.d[indexMV] += tensorDXhat.d[indexXHat]*x_m_mean*(-0.5)*sigmaPow3;
            }
        }
    }

    @Override
    public double getEPS() {
        return EPS;
    }

    @Override
    public void setEPS(double EPS) {
        this.EPS = EPS;
    }

    @Override
    public boolean hasGammaBeta() {
        return requiresGammaBeta;
    }

    @Override
    public Class<Tensor_F64> getTensorType() {
        return Tensor_F64.class;
    }

    @Override
    public Tensor_F64 getMean( Tensor_F64 output ) {
        if( output == null )
            output = tensorMean.createLike();

        output.setTo(tensorMean);

        return output;
    }

    @Override
    public Tensor_F64 getVariance( Tensor_F64 output ) {
        if( output == null )
            output = tensorStd.createLike();

        output.reshape(tensorStd.getShape());

        int indexOut = output.startIndex;
        int indexStd = 0;

        int length = tensorStd.length();

        for (int i = 0; i < length; i++) {
            double d = tensorStd.d[indexStd++];
            output.d[indexOut++] = d*d - EPS;
        }

        return output;
    }
}
