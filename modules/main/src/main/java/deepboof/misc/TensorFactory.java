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

package deepboof.misc;

import deepboof.Tensor;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;

import java.util.Random;

/**
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class TensorFactory<T extends Tensor> {

	Class tensorType;

	public TensorFactory(Class tensorType) {
		this.tensorType = tensorType;
	}

	public T create( int...shape) {
		if( tensorType == Tensor_F64.class ) {
			return (T) new Tensor_F64(shape);
		} else if( tensorType == Tensor_F32.class ) {
			return (T)new Tensor_F32(shape);
		} else {
			throw new IllegalArgumentException("Unknown/unsupported tensor type "+tensorType.getSimpleName());
		}
	}

	/**
	 * Creates a random tensor with the specified shape and values from -1 to 1
	 *
	 * @param rand Random number generator
	 * @param subTensor Should it be a sub-tensor or not?
	 * @param minibatch Number of mini-batches
	 * @param shape Shape of the tensor, without minibatch
	 * @return The random tensor
	 */
	public T randomM(Random rand , boolean subTensor , int minibatch , int shape[] ) {

		int modshape[] = new int[ shape.length + 1];
		modshape[0] = minibatch;
		System.arraycopy(shape,0,modshape,1,shape.length);

		if( tensorType == Tensor_F64.class ) {
			return (T) TensorFactory_F64.random(rand, subTensor, modshape);
		} else if( tensorType == Tensor_F32.class ) {
			return (T) TensorFactory_F32.random(rand,subTensor,modshape);
		} else {
			throw new IllegalArgumentException("Unknown/unsupported tensor type "+tensorType.getSimpleName());
		}
	}

	public T random(Random rand , boolean subTensor , int ...shape ) {
		if( tensorType == Tensor_F64.class ) {
			return (T) TensorFactory_F64.random(rand,subTensor,shape);
		} else if( tensorType == Tensor_F32.class ) {
			return (T) TensorFactory_F32.random(rand,subTensor,shape);
		} else {
			throw new IllegalArgumentException("Unknown/unsupported tensor type "+tensorType.getSimpleName());
		}
	}

	public T random( Random rand , boolean subTensor , double min , double max , int ...shape ) {
		if( tensorType == Tensor_F64.class ) {
			return (T) TensorFactory_F64.randomMM(rand,subTensor,min,max,shape);
		} else if( tensorType == Tensor_F32.class ) {
			return (T) TensorFactory_F32.randomMM(rand,subTensor,(float)min,(float)max,shape);
		} else {
			throw new IllegalArgumentException("Unknown/unsupported tensor type "+tensorType.getSimpleName());
		}
	}

	public Class<T> getTensorType() {
		return tensorType;
	}
}
