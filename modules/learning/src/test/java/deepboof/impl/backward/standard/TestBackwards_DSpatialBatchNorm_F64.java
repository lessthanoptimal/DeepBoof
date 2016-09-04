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

import deepboof.DFunction;
import deepboof.backward.ChecksDerivative;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DSpatialBatchNorm_F64 extends ChecksDerivative<Tensor_F64> {
    private boolean gammaBeta;
    public TestBackwards_DSpatialBatchNorm_F64() {
        numberOfConfigurations = 2;
    }

    @Override
    public DFunction<Tensor_F64> createBackwards(int type) {
        gammaBeta = type == 0;
        return new DSpatialBatchNorm_F64(gammaBeta);
    }

    @Override
    public List<Tensor_F64> createParameters(DFunction<Tensor_F64> function, Tensor_F64 input) {
        if( !gammaBeta )
            return new ArrayList<>();

        Tensor_F64 p = new Tensor_F64( TensorOps.WI(input.length(1),2) );

        for (int i = 0; i < p.d.length; i += 2) {
            p.d[i] = random.nextDouble()*5+0.5;   // gamma
            p.d[i] = (random.nextDouble()-0.5)*4; // beta
        }

        return Arrays.asList(p);
    }

    @Override
    public List<Case> createTestInputs() {

        Case a = new Case(1,1,1);
        a.minibatch = 10;
        Case b = new Case(10,1,1);
        b.minibatch = 20;
        Case c = new Case(10,4,3);
        c.minibatch = 20;

        return Arrays.asList(a,b,c);
    }
}
