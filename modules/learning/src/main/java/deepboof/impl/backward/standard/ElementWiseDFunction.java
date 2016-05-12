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

import deepboof.Tensor;

import java.util.List;

/**
 * Base class for element-wise derivative functions
 *
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public abstract class ElementWiseDFunction<T extends Tensor<T>> extends BaseDFunction<T> {
	@Override
	public void _initialize() {
		shapeOutput = shapeInput.clone();
	}

	@Override
	public void _setParameters(List<T> parameters) {}
}
