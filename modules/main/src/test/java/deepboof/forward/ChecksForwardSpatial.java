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

package deepboof.forward;

import deepboof.Tensor;

import java.util.ArrayList;
import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * Common checks for spatial functions
 *
 * @author Peter Abeles
 */
public abstract class ChecksForwardSpatial<T extends Tensor<T>>
		extends ChecksForward<T> {

	@Override
	public List<int[]> createTestInputs() {
		List<int[]> inputs = new ArrayList<>();

		inputs.add( WI(1,1,1));
		inputs.add( WI(1,5,6));
		inputs.add( WI(3,5,6));
		inputs.add( WI(2,6,5));
		inputs.add( WI(2,12,14));

		return inputs;
	}
}
