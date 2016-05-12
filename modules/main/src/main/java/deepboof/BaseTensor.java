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

package deepboof;

/**
 * Base class for all Tensor data types.  Only provides information dealing with the tensor's shape
 * and core data type.
 *
 * @author Peter Abeles
 */
public abstract class BaseTensor implements ITensor {

	/**
	 * The lengths of each dimension/axis.  Highest stride to smallest stride.
	 */
	public int shape[] = new int[0];

	/**
	 * {@inheritDoc} 
	 */
	public int[] getShape() {
		return shape;
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean isShape(int ...shape ) {
		if( this.shape.length != shape.length )
			return false;
		int N = shape.length-1;
		for (int i = 0; i < shape.length; i++) {
			if( this.shape[i] != shape[i])
				return false;
		}
		return true;
	}

	/**
	 * {@inheritDoc}
	 */
	public int length(int dimension ) {
		if( dimension < 0 )
			return shape[ shape.length+dimension];
		else
			return shape[dimension];
	}

	/**
	 * {@inheritDoc}
	 */
	public int[] createCoor() {
		return new int[ getDimension() ];
	}

	/**
	 * {@inheritDoc}
	 */
	public int getDimension() {
		return shape.length;
	}
}
