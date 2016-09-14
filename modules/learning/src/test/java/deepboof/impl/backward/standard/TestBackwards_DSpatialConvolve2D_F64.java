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
import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.factory.FactoryBackwards;
import deepboof.forward.ConfigConvolve2D;
import deepboof.forward.ConfigPadding;
import deepboof.tensors.Tensor_F64;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DSpatialConvolve2D_F64 extends ChecksDerivative<Tensor_F64> {

    List<ConfigConvolve2D> configuration = new ArrayList<>();
    List<ConfigPadding> configurationPadding = new ArrayList<>();

    int active;

    public TestBackwards_DSpatialConvolve2D_F64() {
        numberOfConfigurations = 2;

        // very simple configuration
        ConfigConvolve2D config = new ConfigConvolve2D();
        config.F = 1;
        config.WW = 1;
        config.HH = 1;

        configuration.add(config);

        ConfigPadding padding = new ConfigPadding();
        padding.x0 = 0;
        padding.y0 = 0;
        padding.x1 = 0;
        padding.y1 = 0;

        configurationPadding.add( padding );

        // more realistic and complex
        config = new ConfigConvolve2D();
        config.F = 4;
        config.WW = 2;
        config.HH = 3;

        configuration.add(config);

        padding = new ConfigPadding();
        padding.x0 = 1;
        padding.y0 = 2;
        padding.x1 = 2;
        padding.y1 = 3;
        configurationPadding.add( padding );
    }

    @Override
    public DFunction<Tensor_F64> createBackwards(int type) {
        this.active = 0;

        FactoryBackwards<Tensor_F64> factory = new FactoryBackwards<>(Tensor_F64.class);

        DSpatialPadding2D_F64 padding = factory.spatialPadding(configurationPadding.get(type));

        return new DSpatialConvolve2D_F64(configuration.get(type),padding);
    }

    @Override
    public List<Tensor_F64> createParameters(DFunction<Tensor_F64> function, Tensor_F64 input) {

        List<Tensor_F64> parameters = new ArrayList<>();

        for( int []shape : function.getParameterShapes() ) {
            parameters.add( tensorFactory.random(random,false,-2.0,2.0,shape));
        }


        return parameters;
    }

    @Override
    public List<Case> createTestInputs() {

        List<Case> inputs = new ArrayList<>();

        inputs.add( new Case(WI(1,1,1)));
        inputs.add( new Case(WI(1,5,6)));
        inputs.add( new Case(WI(3,5,6)));
        inputs.add( new Case(WI(3,12,13)));

        return inputs;
    }
}
