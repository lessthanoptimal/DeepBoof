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
import deepboof.Tensor;
import deepboof.backward.ChecksDerivativeElementWise;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DActivationTanh_F64 extends ChecksDerivativeElementWise {

	@Override
	public DFunction createBackwards(int type) {
		return new DActivationTanH_F64();
	}

	@Override
	public List createParameters(DFunction function, Tensor input) {
		return new ArrayList<>();
	}
}