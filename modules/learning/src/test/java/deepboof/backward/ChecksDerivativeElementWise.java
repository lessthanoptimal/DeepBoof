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

import deepboof.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public abstract class ChecksDerivativeElementWise<T extends Tensor<T>>
	extends ChecksDerivative<T>
{
	@Override
	public List<Case> createTestInputs() {
		List<Case> valid = new ArrayList<>();

		valid.add( new Case());
		valid.add( new Case(1));
		valid.add( new Case(4,1,2));
		valid.add( new Case(2,4,5,2));

		return valid;
	}
}
