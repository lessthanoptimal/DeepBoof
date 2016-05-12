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

import deepboof.DeepBoofConstants;
import deepboof.Function;
import deepboof.backward.NumericalGradient;
import deepboof.misc.TensorOps_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * Implementation of {@link NumericalGradient} for {@link Tensor_F64}
 *
 * @author Peter Abeles
 */
public class NumericalGradient_F64 implements NumericalGradient<Tensor_F64>
{
	Function<Tensor_F64> function;
	// sampling distance
	double T = DeepBoofConstants.TEST_TOL_A_F64;

	Tensor_F64 output = new Tensor_F64();

	// passed in parameters
	Tensor_F64 input;
	List<Tensor_F64> parameters;

	@Override
	public void configure(double T) {

		if( T <= 0 )
			throw new IllegalArgumentException("T must be > 0");

		this.T = T;
	}

	@Override
	public void setFunction(Function<Tensor_F64> function) {
		this.function = function;
	}

	@Override
	public void differentiate(Tensor_F64 input, List<Tensor_F64> parameters, Tensor_F64 dout,
							  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters)
	{
		int N = input.length(0);

		output.reshape( WI(N,function.getOutputShape()) );

		this.input = input;
		this.parameters = parameters;

		process(input,dout,gradientInput);

		for (int i = 0; i < parameters.size(); i++) {
			process(parameters.get(i),dout,gradientParameters.get(i));
		}
	}

	/**
	 * Computes the gradient for a specific tensor
	 */
	private void process( Tensor_F64 target , Tensor_F64 dout , Tensor_F64 gradientTarget ) {

		int length = target.length();

		for (int i = 0; i < length; i++) {
			int indexTarget = target.startIndex + i;
			double v = target.d[indexTarget];

			// value in forward direction
			target.d[indexTarget] = v + T;
			function.setParameters(parameters);
			function.forward(input,output);
			TensorOps_F64.elementMult(output,dout,output);
			double plus_T = TensorOps_F64.elementSum(output);

			// value in backwards direction
			target.d[indexTarget] = v - T;
			function.setParameters(parameters);
			function.forward(input,output);
			TensorOps_F64.elementMult(output,dout,output);
			double minus_T = TensorOps_F64.elementSum(output);

			// undo the changes
			target.d[indexTarget] = v;

			// compute derivative and save the results
			int indexGradient = gradientTarget.startIndex+i;
			gradientTarget.d[indexGradient] = (plus_T-minus_T)/(2.0*T);
		}

	}
}
