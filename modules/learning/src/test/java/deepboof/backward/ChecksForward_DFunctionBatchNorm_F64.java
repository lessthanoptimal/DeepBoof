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
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Special checks for learning implementations of batch norm.  The behavior is very different from forwards
 * only since the statics are computed in the forwards pass
 *
 * @author Peter Abeles
 */
public abstract class ChecksForward_DFunctionBatchNorm_F64 extends ChecksForward_DFunctionBatchNorm<Tensor_F64> {


    public ChecksForward_DFunctionBatchNorm_F64() {
        super(DeepBoofConstants.TEST_TOL_F64);
    }

    @Override
    protected void checkTensorType(Class<Tensor_F64> type) {
        assertTrue( Tensor_F64.class == type );
    }

    @Override
    protected Tensor_F64 createParameter( double gamma , double beta , int shape[] ) {
        Tensor_F64 params = tensorFactory.create(WI(shape,2));
        for (int i = 0; i < params.d.length; i += 2 ) {
            params.d[i] = gamma;
            params.d[i+1] = beta;
        }
        return params;
    }

    @Override
    protected void verifyMean( Tensor_F64 tensor , double expected , double tol ) {
        Tensor_F64 means = tensorFactory.create((TensorOps.outerLength(tensor.shape,1)));

        int numBatch = tensor.length(0);
        int D = means.length();

        int index = 0;
        for (int i = 0; i < numBatch; i++) {
            for (int j = 0; j < D; j++) {
                means.d[j] += tensor.d[index++];
            }
        }
        for (int j = 0; j < D; j++) {
            assertEquals( expected, means.d[j]/numBatch , tol );
        }
    }

    @Override
    protected void verifyStd( Tensor_F64 tensor , double mean, double expected , double tol ) {
        Tensor_F64 stdev = tensorFactory.create(TensorOps.outerLength(tensor.shape,1));

        int numBatch = tensor.length(0);
        int D = stdev.length();

        int index = 0;
        for (int i = 0; i < numBatch; i++) {
            for (int j = 0; j < D; j++) {
                double d = tensor.d[index++]-mean;
                stdev.d[j] += d*d;
            }
        }
        for (int j = 0; j < D; j++) {
            assertEquals( expected, Math.sqrt(stdev.d[j]/(numBatch-1)) , tol );
        }
    }
}
