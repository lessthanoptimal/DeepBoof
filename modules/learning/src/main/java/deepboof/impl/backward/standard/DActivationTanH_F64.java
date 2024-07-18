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

import deepboof.backward.DActivationTanH;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DActivationTanH} for {@link Tensor_F64}. Forward pass tanh is
 * cached to reduce computations in backwards pass.
 *
 * @author Peter Abeles
 */
public class DActivationTanH_F64 extends ElementWiseDFunction<Tensor_F64>
	implements DActivationTanH<Tensor_F64>
{
	// cache tanh computation to avoid doing it more than once
	Tensor_F64 memory = new Tensor_F64();

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {
		memory.reshape(input.getShape());

		int length = input.length();

		int indexDIn = gradientInput.startIndex;
		int indexDout = dout.startIndex;

		for (int i = 0; i < length; i++) {
			double tanh = memory.d[i];
			gradientInput.d[indexDIn++] = (1.0-tanh*tanh)*dout.d[indexDout++];
		}
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		memory.reshape(input.getShape());

		int length = input.length();

		int indexIn = input.startIndex;
		int indexOut = output.startIndex;

		for (int i = 0; i < length; i++) {
			double v = Math.tanh(input.d[indexIn+i]);
			output.d[indexOut+i] = v;
			memory.d[i] = v;
		}
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
