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
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static deepboof.misc.TensorOps.WT;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DFunctionLinear_F64 extends ChecksDerivative<Tensor_F64> {

	@Override
	public DFunction<Tensor_F64> createBackwards(int type) {
		return new DFunctionLinear_F64(7);
	}

	@Override
	public List<Tensor_F64> createParameters(DFunction<Tensor_F64> function, Tensor_F64 input) {
		Tensor_F64 weights = TensorFactory_F64.random(
				random,false,function.getParameterShapes().get(0));
		Tensor_F64 bias = TensorFactory_F64.random(
				random,false,function.getParameterShapes().get(1));

		return WT(weights,bias);
	}

	@Override
	public List<int[]> createTestInputs() {
		List<int[]> valid = new ArrayList<>();

		valid.add( WI(1));
		valid.add( WI(1,1));
		valid.add( WI(4,1,2));
		valid.add( WI(2,4,5,2));

		return valid;
	}
}
