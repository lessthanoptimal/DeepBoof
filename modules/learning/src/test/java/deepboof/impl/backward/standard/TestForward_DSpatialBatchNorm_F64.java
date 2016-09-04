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

import deepboof.backward.ChecksForward_DSpatialBatchNorm_F64;
import deepboof.backward.DSpatialBatchNorm;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
public class TestForward_DSpatialBatchNorm_F64 extends ChecksForward_DSpatialBatchNorm_F64 {

    @Override
    public DSpatialBatchNorm<Tensor_F64> createForwards(boolean gammaBeta) {
        return new DSpatialBatchNorm_F64( gammaBeta );
    }
}
