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

import deepboof.forward.FunctionLinear;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * Implementation of {@link FunctionLinear} for {@link Tensor_F64}.
 *
 * @author Peter Abeles
 */
public class FunctionLinear_F64 extends BaseFunction<Tensor_F64>
		implements FunctionLinear<Tensor_F64> {

	// number of inputs
	protected int D;
	// number of outputs
	protected int M;

	Tensor_F64 weight;
	Tensor_F64 bias;

	public FunctionLinear_F64(int numberOfOutputs) {
		M = numberOfOutputs;
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		forwards(input, output, weight, bias, miniBatchSize, D, M);
	}

	public static void forwards(Tensor_F64 input, Tensor_F64 output,
								 Tensor_F64 weight, Tensor_F64 bias,
								int miniBatchSize, int D, int M)
	{
		// See FunctionLinear for complete documentation
		// Input   = (N,d[1], ... , d[K])
		// Weights = (M,D)
		// Bias    = (M)
		// Output  = (N,M)

		for (int stack = 0; stack < miniBatchSize; stack++) {
			int indexStartIn = stack* D + input.startIndex;

			// perform matrix multiplication, note how the input and weight shape has been selected so that
			// a simple for loop is all that is needed.
			// Also, remember that tensors are in row major format, which is why the input can be treated
			// as a continuous array here
			for (int outputElement = 0; outputElement < M; outputElement++) {
				int indexW = outputElement* D + weight.startIndex;

				double b = bias.d[outputElement + bias.startIndex];

				int indexIn = indexStartIn;
				int end = indexIn + D;
				double sum = 0;
				while( indexIn < end ) {
					sum += input.d[indexIn++]*weight.d[indexW++];
				}

				int indexOut = stack* M + outputElement + output.startIndex;
				output.d[indexOut] = sum + b;
			}
		}
	}

	@Override
	public void _initialize() {
		if( shapeInput.length < 1 ) {
			throw new IllegalArgumentException("Input tensor shape must have a dimension of at least 1");
		}
		// compute number of inputs, which is a volume
		D = TensorOps.tensorLength(shapeInput);

		// shape of weights
		shapeParameters.add( new int[]{M,D});
		// shape of biases
		shapeParameters.add( new int[]{M});

		// shape of output
		shapeOutput = new int[]{M};
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {
		weight = parameters.get(0);
		bias = parameters.get(1);
	}

	@Override
	public int getNumberOfOutputs() {
		return D;
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
