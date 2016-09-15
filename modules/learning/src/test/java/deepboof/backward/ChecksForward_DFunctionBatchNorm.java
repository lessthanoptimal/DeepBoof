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

package deepboof.backward;

import deepboof.DeepBoofConstants;
import deepboof.DeepUnitTest;
import deepboof.Tensor;
import deepboof.misc.TensorOps;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Special checks for learning implementations of batch norm.  The behavior is very different from forwards
 * only since the statics are computed in the forwards pass
 *
 * @author Peter Abeles
 */
public abstract class ChecksForward_DFunctionBatchNorm<T extends Tensor<T>>
        extends ChecksForward_DBatchNorm<T> {

    public ChecksForward_DFunctionBatchNorm(double tolerance) {
        super(tolerance);
    }

    public abstract DFunctionBatchNorm<T> createForwards(boolean gammaBeta );

    @Override
    public DFunctionBatchNorm<T> createForwards(int which) {
        gammaBeta = which == 0;
        return createForwards(gammaBeta);
    }

    @Override
    protected void checkParameterShapes(int[] input, List<int[]> parameters) {
        if( gammaBeta ) {
            assertEquals(1, parameters.size());
            int[] paramShape = parameters.get(0);

            assertEquals(input.length + 1, paramShape.length);

            for (int i = 0; i < input.length; i++) {
                assertEquals(input[i], paramShape[i]);
            }
            assertEquals(2, paramShape[paramShape.length - 1]);
        } else {
            assertEquals(0,parameters.size());
        }
    }

    @Override
    protected void checkOutputShapes(int[] input, int[] output) {
        DeepUnitTest.assertEquals(input, output);
    }

    @Override
    public List<Case> createTestInputs() {

        List<Case> cases = new ArrayList<Case>();

        cases.add( new Case(10));
        cases.add( new Case(3,4,5));

        // do a bunch of mini-batches so that the statistics are meaningful
        for( Case c : cases )
            c.minibatch = 200;

        return cases;
    }

    /**
     * Ensures that the mean and variance of the output is approximately one and also checks the statistics of
     * the internal
     */
    @Test
    public void checkOutputStatistics() {
        for( int config = 0; config < numberOfConfigurations; config++ ) {
            DFunctionBatchNorm<T> alg = createForwards(config);
            alg.learning();

            for (Case test : createTestInputs()) {
                T input = tensorFactory.randomM(random,false,test.minibatch,test.inputShape);
                T output = tensorFactory.randomM(random,false,test.minibatch,test.inputShape);

                alg.initialize(test.inputShape);
                if( alg.hasGammaBeta() ) {
                    // declare the parameters such that they will not shift the output
                    T params = createParameter(1,0,test.inputShape);
                    alg.setParameters(Arrays.asList(params));
                }
                alg.forward(input,output);

                verifyMean(output, 0, DeepBoofConstants.TEST_TOL_B_F64);
                verifyStd(output, 0, 1.0,  DeepBoofConstants.TEST_TOL_B_F64);

                // the data is generated from a uniform distribution from -1 to 1
                // mean should be 0 and variance 4/12 = 0.333
                T foundMean =  alg.getMean(null);
                T foundVariance = alg.getVariance(null);

                double foundMeanAve = TensorOps.elementSum(foundMean)/foundMean.length();
                double foundVarAve = TensorOps.elementSum(foundVariance)/foundVariance.length();

                // standard deviation of the mean
                double sampleMeanStd = 0.333/Math.sqrt(test.minibatch);

                // the mean should be within this tolerance almost all of the time
                assertEquals(0, foundMeanAve, sampleMeanStd*3);
                assertEquals(0.333, foundVarAve , sampleMeanStd*10); // not mathematically sound tolerance
            }
        }
    }

    /**
     * Tests to see if gamma and beta are applied to the output correctly
     */
    @Test
    public void checkGammaBeta() {
        DFunctionBatchNorm<T> alg = createForwards(true);
        alg.learning();

        assertTrue( alg.hasGammaBeta() );

        int shape[] = new int[]{20};

        T input = tensorFactory.randomM(random,false,30,shape);
        T output = tensorFactory.randomM(random,false,30,shape);

        T params = createParameter(1.5,20,shape);
        alg.initialize(shape);
        alg.setParameters(Arrays.asList(params));
        alg.forward(input,output);
        verifyMean(output, 20 , DeepBoofConstants.TEST_TOL_B_F64);
        verifyStd(output, 20, 1.5,  DeepBoofConstants.TEST_TOL_B_F64);
    }

    protected abstract T createParameter( double gamma , double beta , int shape[] );

    protected abstract void verifyMean( T tensor , double expected , double tol );

    protected abstract void verifyStd( T tensor , double mean, double expected , double tol );
}
