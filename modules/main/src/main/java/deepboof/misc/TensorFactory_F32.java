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

import deepboof.tensors.Tensor_F32;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Various functions for unit tests
 *
 * @author Peter Abeles
 */
public class TensorFactory_F32 {

	/**
	 * Generate a zeros tensor with the option for a sub-tensor
	 *
	 * @param rand If you wish to generate a sub-matrix pass in this RNG and it will randomly offset the data.  null
	 *             for regular tensor
	 * @param shape Shape of the tensor
	 * @return tensor
	 */
	public static Tensor_F32 zeros( Random rand, int ...shape ) {
		Tensor_F32 out = new Tensor_F32();

		if( rand != null ) {
			out.subtensor = true;
			out.startIndex = rand.nextInt(20)+1;
		}

		out.d = new float[ out.startIndex + TensorOps.tensorLength(shape)];
		out.reshape(shape);
		return out;
	}

	/**
	 * Creates a random tensor with the specified shape and values from -1 to 1
	 *
	 * @param rand Random number generator
	 * @param subTensor Should it be a sub-tensor or not?
	 * @param shape Shape of the tensor
	 * @return The random tensor
	 */
	public static Tensor_F32 random(Random rand , boolean subTensor , int ...shape ) {
		return randomMM(rand,subTensor,-1.0f, 1.0f, shape);
	}

	/**
	 * Creates a random tensor with the specified shape and value range
	 *
	 * @param rand Random number generator
	 * @param subTensor Should it be a sub-tensor or not?
	 * @param min Minimum value of each element
	 * @param max Maximum value of each element
	 * @param shape Shape of the tensor
	 * @return The random tensor
	 */
	public static Tensor_F32 randomMM( Random rand , boolean subTensor , float min , float max , int ...shape ) {
		Tensor_F32 out = zeros(subTensor?rand:null,shape);

		randomMM(rand,min,max,out);

		return out;
	}

	/**
	 * Creates a random tensor with the specified shape and value range
	 *
	 * @param rand Random number generator
	 * @param subTensor Should it be a sub-tensor or not?
	 * @param min Minimum value of each element
	 * @param max Maximum value of each element
	 * @param shapes Shapes of the tensors
	 * @return The random tensor
	 */
	public static List<Tensor_F32> randomMM(Random rand , boolean subTensor , float min , float max , List<int[]> shapes ) {

		List<Tensor_F32> out = new ArrayList<Tensor_F32>();

		for( int[]shape : shapes ) {
			out.add( randomMM(rand,subTensor,min,max,shape));
		}

		return out;
	}

	/**
	 * Fills the tensor with random numbers selected from a uniform distribution.
	 *
	 * @param rand Random number generator
	 * @param min min value, inclusive
	 * @param max max value, inclusive
	 * @param tensor Tensor that is to be filled.
	 */
	public static void randomMM( Random rand , float min , float max , Tensor_F32 tensor ) {
		int N = tensor.length();
		for (int i = 0; i < N; i++) {
			tensor.d[ tensor.startIndex + i ] = rand.nextFloat()*(max-min) + min;
		}
	}
}
