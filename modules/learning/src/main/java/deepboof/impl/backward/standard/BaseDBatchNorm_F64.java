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
import deepboof.backward.DBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implements common functionality for all batch normalization implementations for {@link Tensor_F64}.
 *
 * @author Peter Abeles
 */
public abstract class BaseDBatchNorm_F64 extends BaseDFunction<Tensor_F64> implements DBatchNorm<Tensor_F64>
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

    // x[i] - mean(x)
    protected Tensor_F64 tensorDiffX = new Tensor_F64();

    // temporary storage
    protected Tensor_F64 tensorTmp = new Tensor_F64();

    // shape of a tensor that just contains the variables
    protected int[] shapeVariables;

    // number of elements in input tensor (excluding mini-batch)
    protected int D;

    // Internal storage for gamma and beta parameters.  Stored interleaved gamma then beta.  1 for each input variable
    // params = [ gamma[0], beta[0], gamma[1], beta[1],  ... , gamma[D], beta[D]]
    protected Tensor_F64 params = new Tensor_F64(0);
    protected double EPS = DeepBoofConstants.TEST_TOL_F64*0.1;

    public BaseDBatchNorm_F64(boolean requiresGammaBeta) {
        this.requiresGammaBeta = requiresGammaBeta;
    }

    @Override
    public void _initialize() {
        shapeVariables = createShapeVariables(shapeInput);

        tensorMean.reshape(shapeVariables);
        tensorStd.reshape(shapeVariables);
        tensorDVar.reshape(shapeVariables);
        tensorDMean.reshape(shapeVariables);
        tensorTmp.reshape(shapeVariables);

        this.shapeOutput = shapeInput.clone();

        if( requiresGammaBeta ) {
            int shapeParam[] = TensorOps.WI(shapeVariables, 2);
            this.shapeParameters.add(shapeParam);
            params.reshape(shapeParam);
        }

        D = TensorOps.tensorLength(shapeVariables);
    }

    /**
     * Create the shape for all the variables which are being normalized
     * @param shapeInput Shape of input tensor (without mini-batch)
     * @return shape of variables tensor
     */
    protected abstract int[] createShapeVariables(  int shapeInput[] );

    @Override
    public void _setParameters(List<Tensor_F64> parameters) {
        if( requiresGammaBeta ) {
            params.setTo(parameters.get(0));
        } else if( parameters.size() != 0 ){
            throw new IllegalArgumentException("There are no parameters since gamma and beta have been turned off");
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
