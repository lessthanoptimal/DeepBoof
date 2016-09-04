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
import deepboof.backward.DSpatialBatchNorm;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DSpatialBatchNorm}.  Inside it uses {@link DFunctionBatchNorm} by reordering the input
 * such that all the elements inside a single band is converted into a mini-batch then converts it back.
 *
 * @author Peter Abeles
 */
public class DSpatialBatchNorm_F64 extends BaseDFunction<Tensor_F64> implements DSpatialBatchNorm<Tensor_F64> {
    DFunctionBatchNorm_F64 innerAlg;

    Tensor_F64 flattenedIn = new Tensor_F64();
    Tensor_F64 flattenedOut = new Tensor_F64();

    public DSpatialBatchNorm_F64( boolean requiresGammaBeta ) {
        innerAlg = new DFunctionBatchNorm_F64(requiresGammaBeta);
    }


    @Override
    public void _initialize() {
        if( shapeInput.length != 3 )
            throw new IllegalArgumentException("Expected 3 DOF in a spatial shape (C,W,H)");
        this.shapeOutput = shapeInput.clone();

        // When flattenedIn each channel is treated as a variable
        int flatShape[] = new int[]{shapeInput[0]};

        if( innerAlg.hasGammaBeta() ) {
            int shapeParam[] = new int[]{shapeInput[0], 2};
            this.shapeParameters.add(shapeParam);
        }

        innerAlg.initialize(flatShape);
    }

    @Override
    public void _setParameters(List<Tensor_F64> parameters) {
        innerAlg.setParameters(parameters);
    }

    @Override
    public void _forward(Tensor_F64 input, Tensor_F64 output) {
        int C = input.length(1); // number of input channels
        int numMiniBatches = input.length()/C; // after reshaping this is the number of mini batches it will have

        // flatten the input spatial tensor
        flattenedIn.reshape(numMiniBatches, C);
        spatialToFunction(input, flattenedIn);

        // process it
        flattenedOut.reshape(numMiniBatches, C);
        innerAlg.forward(flattenedIn,flattenedOut);

        // turn it back into a spatial tensor
        functionToSpatial(flattenedOut,output);
    }

    @Override
    protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {
        int C = input.length(1); // number of input channels
        int numMiniBatches = input.length()/C; // after reshaping this is the number of mini batches it will have

        // flatten the input spatial tensor
        flattenedIn.reshape(numMiniBatches, C);
        spatialToFunction(input, flattenedIn);

        // process it
        flattenedOut.reshape(numMiniBatches, C);
//        innerAlg.backwards(flattenedIn,flat_dout,flat_grad, gradientParameters);

        // turn it back into a spatial tensor
//        functionToSpatial(flattenedOut,output);
    }

    /**
     * <p>Reorders planar to format expected by function batch norm.</p>
     *
     * Example of an input with just 1 minibatch and two 3x3 planes, into an output with two variables
     * and 9 mini-batches
     * <pre>
     * [ 012 , 345 ]     [ 012678234 ]
     * [ 678 , 901 ]  -> [ 345901567 ]
     * [ 234 , 567 ]
     * </pre>
     *
     * @param inputSpatial Spatial tensor with 4 dimensions
     * @param outputFunc Tensor with 2 dimensions.
     */
    public static void spatialToFunction( Tensor_F64 inputSpatial , Tensor_F64 outputFunc ) {
        // number of mini-batches, channels, and pixels in input tensor
        int miniBatchSize = inputSpatial.length(0);
        int C = inputSpatial.length(1);
        int D = TensorOps.outerLength(inputSpatial.shape,2);

        // traverse through input tensor in row-major order while writing to output tensor out of order
        for (int batch = 0; batch < miniBatchSize; batch++) {
            int indexIn = inputSpatial.startIndex;
            for( int channel = 0; channel < C; channel++ ) {

                for (int pixel = 0; pixel < D; pixel++) {
                    int outBatch = batch*D+pixel; // which mini-batch in the outpt is it working on now
                    // each mini-batch is composed of C variables
                    int indexOut = outBatch*C + channel; // tensor is declared internally and starts at index 0

                    outputFunc.d[ indexOut ] = inputSpatial.d[ indexIn ];
                }
            }
        }
    }

    /**
     * Inverse of {@link #spatialToFunction(Tensor_F64, Tensor_F64)}
     */
    public static void functionToSpatial( Tensor_F64 inputFunc , Tensor_F64 outputSpatial ) {
        int miniBatchSize = outputSpatial.length(0);
        int C = outputSpatial.length(1);
        int D = TensorOps.outerLength(outputSpatial.shape,2);

        for (int batch = 0; batch < miniBatchSize; batch++) {
            int indexOut = outputSpatial.startIndex;
            for( int channel = 0; channel < C; channel++ ) {

                for (int pixel = 0; pixel < D; pixel++) {
                    int inBatch = batch*D+pixel;
                    int indexIn = inBatch*C + channel;

                    outputSpatial.d[ indexOut ] = inputFunc.d[ indexIn ];
                }
            }
        }
    }

    @Override
    public Class<Tensor_F64> getTensorType() {
        return innerAlg.getTensorType();
    }

    @Override
    public boolean hasGammaBeta() {
        return innerAlg.hasGammaBeta();
    }

    @Override
    public double getEPS() {
        return innerAlg.getEPS();
    }

    @Override
    public void setEPS(double EPS) {
        this.innerAlg.setEPS(EPS);
    }
}
