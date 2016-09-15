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

import deepboof.backward.DFunctionDropOut;
import deepboof.misc.TensorOps_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;
import java.util.Random;

/**
 * Implementation of {@link DFunctionDropOut} for {@link Tensor_F64}
 *
 * @author Peter Abeles
 */
public class DFunctionDropOut_F64 extends BaseDFunction<Tensor_F64> implements DFunctionDropOut<Tensor_F64> {

	Random random;

	// Specifies chance of a neuron being dropped from 0 to 1.0
	double dropRate;

	// used to indicate if a neuron is turned off or not.  Using a double since it should be faster
	// than adding conditional statements (need to verify this)
	Tensor_F64 drops = new Tensor_F64();

	/**
	 * Configures drop out
	 * @param randomSeed random seed used to pick which neurons are dropped
	 * @param dropRate Fraction of time a neuron is dropped
	 */
	public DFunctionDropOut_F64( long randomSeed , double dropRate) {
		this.random = new Random(randomSeed);
		this.dropRate = dropRate;
	}

	@Override
	public void _initialize() {
		shapeOutput = shapeInput.clone();
	}

	@Override
	public void _setParameters(List<Tensor_F64> parameters) {}

	@Override
	public void _forward(Tensor_F64 input, Tensor_F64 output) {
		if( learningMode ) {
			drops.reshape(input.shape);
			int N = drops.length();

			int indexIn = input.startIndex;
			int indexOut = output.startIndex;

			for (int i = 0; i < N; i++) {
				double d = drops.d[i] = random.nextDouble() < dropRate ? 0.0 : 1.0;
				output.d[indexOut++] = input.d[indexIn++]*d;
			}
		} else {
			TensorOps_F64.elementMult(input,1.0-dropRate,output);
		}
	}

	@Override
	public double getDropRate() {
		return dropRate;
	}

	@Override
	protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {
		TensorOps_F64.elementMult(dout,drops,gradientInput);
	}

	@Override
	public Class<Tensor_F64> getTensorType() {
		return Tensor_F64.class;
	}
}
