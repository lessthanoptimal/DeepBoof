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

import deepboof.backward.DActivationSigmoid;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link DActivationSigmoid} for {@link Tensor_F64}.  Saves the sigmoid computed
 * on the forward pass to avoid recomputing the sigmoid on the backwards pass.
 *
 * @author Peter Abeles
 */
public class DActivationSigmoid_F64 extends ElementWiseDFunction<Tensor_F64>
		implements DActivationSigmoid<Tensor_F64> {

	// storage for the previously computed sigmoid results
	Tensor_F64 memorySigmoid = new Tensor_F64();

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {

		memorySigmoid.reshape(input.shape);

		int length = input.length();

		int indexIn = input.startIndex;
		int indexOut = output.startIndex;
		int indexMem = memorySigmoid.startIndex;

		for (int i = 0; i < length; i++) {
			double value = input.d[indexIn+i];

			// compute and save the sigmoid for each element
			memorySigmoid.d[indexMem+i] = output.d[indexOut+i] = 1.0/(1.0 + Math.exp(-value));
		}
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

		int length = gradientInput.length();

		int indexDIn = gradientInput.startIndex;
		int indexDOut = dout.startIndex;

		for (int i = 0; i < length; i++) {
			double sigmoid = memorySigmoid.d[i];

			// the sigmoid derivative can be computed using the original sigmoid
			gradientInput.d[indexDIn++] = sigmoid*(1.0-sigmoid)*dout.d[indexDOut++];
		}
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
