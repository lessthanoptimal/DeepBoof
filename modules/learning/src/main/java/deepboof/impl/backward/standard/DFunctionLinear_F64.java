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

import deepboof.backward.DFunctionLinear;
import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

import static deepboof.impl.forward.standard.FunctionLinear_F64.forwards;

/**
 * Implementation of {@link DFunctionLinear} for {@link Tensor_F64}
 *
 * @author Peter Abeles
 */
public class DFunctionLinear_F64 extends BaseDFunction<Tensor_F64>
	implements DFunctionLinear<Tensor_F64>
{
	// number of inputs
	protected int D;
	// number of outputs
	protected int M;

	Tensor_F64 weight;
	Tensor_F64 bias;

	public DFunctionLinear_F64(int numberOfOutputs) {
		M = numberOfOutputs;
	}

	@Override
	public int getNumberOfOutputs() {
		return M;
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {
		weight = parameters.get(0);
		bias = parameters.get(1);
	}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		forwards(input, output, weight, bias, miniBatchSize, D, M);
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {
		// See FunctionLinear for complete documentation
		// Input   = (N,d[1], ... , d[K])
		// Weights = (M,D)
		// Bias    = (M)
		// Output  = (N,M)

		Tensor_F64 inputD = gradientInput;
		Tensor_F64 weightD = gradientParameters.get(0);
		Tensor_F64 biasD = gradientParameters.get(1);

		inputD.zero();
		weightD.zero();
		biasD.zero();

		for (int stack = 0; stack < miniBatchSize; stack++) {
			for (int outputElement = 0; outputElement < M; outputElement++) {
				int indexW = outputElement*D + weight.startIndex;
				int indexX = stack* D + input.startIndex;

				double val_dout = dout.get(stack,outputElement);

				// compute gradient of input tensor and weight
				int indexXD = stack*D + inputD.startIndex;
				int indexWD = outputElement*D + weightD.startIndex;
				for (int i = 0; i < D; i++) {
					inputD.d[indexXD++] += weight.d[indexW+i]*val_dout;
					weightD.d[indexWD++] += input.d[indexX+i]*val_dout;
				}

				// gradient of bias
				biasD.d[biasD.startIndex+outputElement] += val_dout;
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
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
