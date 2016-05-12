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

import deepboof.backward.DActivationReLU;
import deepboof.impl.forward.standard.ActivationReLU_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DActivationReLU} for {@link Tensor_F64}
 *
 * @author Peter Abeles
 */
public class DActivationReLU_F64 extends ElementWiseDFunction<Tensor_F64> implements DActivationReLU<Tensor_F64>{

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		ActivationReLU_F64._relu_forwards(input, output);
	}

	@Override
	protected void _backwards(Tensor_F64 input,
							  Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

		int length = gradientInput.length();

		int indexIn = input.startIndex;
		int indexDIn = gradientInput.startIndex;
		int indexDOut = dout.startIndex;

		for (int i = 0; i < length; i++) {
			double value = input.d[indexIn+i];
			if( value <= 0 )
				gradientInput.d[indexDIn+i] = 0;
			else
				gradientInput.d[indexDIn+i] = dout.d[indexDOut+i];
		}
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
