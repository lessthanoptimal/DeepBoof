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

import deepboof.Function;
import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.factory.FactoryBackwards;
import deepboof.forward.ChecksForwardSpatialConvolve2D_F64;
import deepboof.forward.ConfigConvolve2D;
import deepboof.forward.ConfigPadding;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
public class TestForward_DSpatialConvolve2D_F64 extends ChecksForwardSpatialConvolve2D_F64 {

    @Override
    protected Function<Tensor_F64> createForwards(ConfigConvolve2D configConv, ConfigPadding configPadding) {
        FactoryBackwards<Tensor_F64> factory = new FactoryBackwards<>(Tensor_F64.class);

        DSpatialPadding2D_F64 padding = factory.spatialPadding(configPadding);

        return new DSpatialConvolve2D_F64(configConv,padding);
    }
}
