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

import deepboof.DFunction;
import deepboof.Tensor;
import deepboof.impl.forward.standard.BaseFunction;
import deepboof.misc.TensorOps;

import java.util.List;

/**
 * Base class which implements common functionality between all {@link DFunction}
 *
 * @author Peter Abeles
 */
public abstract class BaseDFunction<T extends Tensor<T>>
		extends BaseFunction<T>
		implements DFunction<T>
{
	protected boolean learningMode = false;

	@Override
	public void learning() {
		learningMode = true;
	}

	@Override
	public void evaluating() {
		learningMode = false;
	}

	@Override
	public void backwards(T input, T dout,
						  T gradientInput, List<T> gradientParameters) {

		if( shapeInput == null )
			throw new IllegalArgumentException("Must initialize first!");
		if( !learningMode )
			throw new IllegalArgumentException("Must be in learning mode ot invoke backwards");

		TensorOps.checkShape("input",-1,shapeInput,input.getShape(),true);

		TensorOps.checkShape("dout", -1, shapeOutput, dout.getShape(),true);
		TensorOps.checkShape("gradientInput",-1,shapeInput,gradientInput.getShape(),true);
		TensorOps.checkShape("gradientParameters",shapeParameters,(List)gradientParameters,false);

		_backwards(input,dout,gradientInput,gradientParameters);
	}

	protected abstract void _backwards(T input, T dout, T gradientInput, List<T> gradientParameters);

	@Override
	public boolean isLearning() {
		return learningMode;
	}
}
