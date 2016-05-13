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

import deepboof.Function;
import deepboof.Tensor;
import deepboof.misc.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Base class which implements common functionality between all {@link Function functions}
 *
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public abstract class BaseFunction<T extends Tensor> implements Function<T> {
	protected int [] shapeInput = new int[0];
	protected List<int []> shapeParameters = new ArrayList<>();
	protected int [] shapeOutput = new int[0];

	protected List<T> parameters;

	/**
	 * Number of inputs in the mini-batch
	 */
	protected int miniBatchSize;

	@Override
	public void initialize(int... shapeInput) {
		this.shapeInput = shapeInput.clone();
		shapeParameters.clear();
		Arrays.fill(shapeOutput,-1);

		_initialize();
	}

	public abstract void _initialize();

	@Override
	public void setParameters(List<T> parameters) {
		TensorOps.checkShape("parameters", shapeParameters, (List) parameters, false);

		this.parameters = new ArrayList<>(parameters);
		_setParameters(parameters);
	}

	public abstract void _setParameters(List<T> parameters);

	@Override
	public List<T> getParameters() {
		return parameters;
	}

	@Override
	public void forward(T input, T output) {
		if( shapeInput == null )
			throw new IllegalArgumentException("Must initialize first!");

		TensorOps.checkShape("input",-1,shapeInput,input.getShape(),true);
		TensorOps.checkShape("output", -1,shapeOutput,output.getShape(),true);

		// see if the number of stacked inputs is the same in input and output
		miniBatchSize = input.length(0);
		if( output.length(0) != miniBatchSize) {
			int M = output.length(0);
			throw new IllegalArgumentException("Dimension 0 in the output is "+M+
					" and does not match input dimension 0 of "+ miniBatchSize);
		}

		_forward(input, output);
	}

	public abstract void _forward(T input, T output);

	@Override
	public List<int[]> getParameterShapes() {
		return shapeParameters;
	}

	@Override
	public int[] getOutputShape() {
		return shapeOutput;
	}
}
