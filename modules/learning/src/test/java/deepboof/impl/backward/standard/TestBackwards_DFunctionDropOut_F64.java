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
import deepboof.tensors.Tensor_F64;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DFunctionDropOut_F64 extends ChecksDerivative<Tensor_F64> {

	@Override
	public DFunction<Tensor_F64> createBackwards(int type) {
		return new DFunctionDropOut_F64(0xdeadbeef,0.3);
	}

	@Override
	public List<Tensor_F64> createParameters(DFunction<Tensor_F64> function, Tensor_F64 input) {
		return new ArrayList<>();
	}

	@Override
	public List<Case> createTestInputs() {
		Case a = new Case();
		a.inputShape = new int[]{4,3,2};
		a.minibatch = 3;

		Case b = new Case();
		b.inputShape = new int[]{5};
		b.minibatch = 1;

		return Arrays.asList(a,b);
	}
}
