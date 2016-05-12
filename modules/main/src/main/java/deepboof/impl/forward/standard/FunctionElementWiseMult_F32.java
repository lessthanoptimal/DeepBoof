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

package deepboof.impl.forward.standard;

import deepboof.forward.FunctionElementWiseMult;
import deepboof.tensors.Tensor_F32;

/**
 * Implementation of {@link FunctionElementWiseMult} for {@link Tensor_F32}.
 *
 * @author Peter Abeles
 */
public class FunctionElementWiseMult_F32
		extends ElementWiseFunction<Tensor_F32>
		implements FunctionElementWiseMult<Tensor_F32>
{
	float scalar;

	public FunctionElementWiseMult_F32(float scalar) {
		this.scalar = scalar;
	}

	@Override
	public void _forward(Tensor_F32 input, Tensor_F32 output) {
		int indexIn = input.startIndex;
		int indexOut = output.startIndex;
		int end = indexIn + input.length();

		while( indexIn < end ) {
			output.d[indexOut++] = scalar*input.d[indexIn++];
		}
	}

	@Override
	public Class<Tensor_F32> getTensorType() {
		return Tensor_F32.class;
	}

	public float getScalar() {
		return scalar;
	}
}
