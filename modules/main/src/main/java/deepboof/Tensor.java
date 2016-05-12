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

import deepboof.misc.TensorOps;

/**
 * Base class for Tensors.  A tensor is an N-dimensional array.  Array elements are stored in a 1-D array
 * in row-major ordering.  This class and its direct children provide only a light weight wrapper around the
 * array.
 *
 * <p>
 * Sub-tensors are tensors which are wrapped around an external array which it doesn't own.  Most of the time,
 * sub-tensors will have a non-zero startIndex.  They can't be {@link #reshape(int...) reshaped} if the change
 * in shape requires the data array to grow.
 * </p>
 *
 * @author Peter Abeles
 */
public abstract class Tensor<T extends Tensor> extends BaseTensor {

	/**
	 * The index in the input array that this tensor starts at.  This allows for a tensor to be
	 * inside a much larger array.
	 */
	public int startIndex = 0;

	/**
	 * If this tensor is wrapped around another array which it doesn't own then it is a sub-tensor.
	 * Most of the time, sub-tensors will have a non-zero startIndex.  They can't be resized.
	 */
	public boolean subtensor = false;

	/**
	 * Stride for each axis
	 */
	public int strides[] = new int[0];

	/**
	 * Accessor function which allows any tensor's element to be read as a double.
	 * @param coordinate Coordinate of the element which is to be read
	 * @return Tensor elements's value as a double
	 */
	public abstract double getDouble( int ...coordinate );

	/**
	 * Returns internal array used to store tensor data.  Data is stored in a row-major order in a single array.
	 * @return tensor data.
	 */
	public abstract Object getData();

	/**
	 * Used to change the internal array in an abstract way
	 */
	public abstract void setData( Object data );

	/**
	 * Reshape for an arbitrary number of dimensions
	 *
	 * @param shape New shape.  Length must be the same as the number of dimensions. Array reference
	 *              not saved internally. Highest to lowest dimension
	 */
	public void reshape(int... shape) {

		if( this.shape.length != shape.length ) {
			this.shape = new int[shape.length];
		}

		System.arraycopy(shape,0,this.shape,0,shape.length);
		reshape();
	}

	/**
	 * {@link #reshape(int...) Reshape} for 1-D tensors.  Convenience function, but can be used
	 * to avoid declaring a new array
	 *
	 * @param length0 Length of axis-0
	 */
	public void reshape( int length0 ) {
		if( shape.length != 1 ) {
			shape = new int[1];
		}
		shape[0] = length0;

		reshape();
	}

	/**
	 * {@link #reshape(int...) Reshape} for 2-D tensors.  Convenience function, but can be used
	 * to avoid declaring a new array
	 *
	 * @param length1 Length of axis-1
	 * @param length0 Length of axis-0
	 */
	public void reshape( int length1 , int length0) {
		if( shape.length != 2 ) {
			shape = new int[2];
		}
		shape[0] = length1;
		shape[1] = length0;

		reshape();
	}

	/**
	 * {@link #reshape(int...) Reshape} for 3-D tensors.  Convenience function, but can be used
	 * to avoid declaring a new array
	 *
	 * @param length2 Length of axis-2
	 * @param length1 Length of axis-1
	 * @param length0 Length of axis-0
	 */
	public void reshape( int length2 , int length1 , int length0 ) {
		if( shape.length != 3 ) {
			shape = new int[3];
		}
		shape[0] = length2;
		shape[1] = length1;
		shape[2] = length0;

		reshape();
	}

	/**
	 * {@link #reshape(int...) Reshape} for 4-D tensors.  Convenience function, but can be used
	 * to avoid declaring a new array
	 *
	 * @param length3 Length of axis-3
	 * @param length2 Length of axis-2
	 * @param length1 Length of axis-1
	 * @param length0 Length of axis-0
	 */
	public void reshape( int length3 , int length2 , int length1 , int length0 ) {
		if( shape.length != 4 ) {
			shape = new int[4];
		}
		shape[0] = length3;
		shape[1] = length2;
		shape[2] = length1;
		shape[3] = length0;

		reshape();
	}

	/**
	 * {@link #reshape(int...) Reshape} for 5-D tensors.  Convenience function, but can be used
	 * to avoid declaring a new array
	 *
	 * @param length4 Length of axis-4
	 * @param length3 Length of axis-3
	 * @param length2 Length of axis-2
	 * @param length1 Length of axis-1
	 * @param length0 Length of axis-0
	 */
	public void reshape( int length4 , int length3 , int length2 , int length1 , int length0 ) {
		if( shape.length != 5 ) {
			shape = new int[5];
		}
		shape[0] = length4;
		shape[1] = length3;
		shape[2] = length2;
		shape[3] = length1;
		shape[4] = length0;

		reshape();
	}

	/**
	 * Reshape for when the inner shape variable has already been adjusted.  Useful for when calls to new
	 * are being minimized.
	 */
	public void reshape() {
		int N = TensorOps.tensorLength(shape);
		computeStrides();

		if( innerArrayLength() < N+startIndex ) {
			if( subtensor )
				throw new IllegalArgumentException("Can't reshape sub-tensors if it requires the data array to grow");
			else if( startIndex != 0 )
				throw new RuntimeException("BUG: Not a sub-tensor and startIndex isn't zero!");
			innerArrayGrow(N);
		}
	}

	/**
	 * Re-declare inner array so that it is at least of length N
	 * @param N Desired minimum length of inner array
	 */
	protected abstract void innerArrayGrow(int N );

	/**
	 * Length of inner array as returned by "data.length"
	 * @return Length of inner array
	 */
	protected abstract int innerArrayLength();

	/**
	 * Returns the index of the coordinate.  Data array is encoded in a row-major format
	 *
	 * @param coordinate Coordinate from highest to lowest axis number/dimension
	 * @return index of the index in internal data array
	 */
	public int idx(int ...coordinate ) {
		int index = 0;
		for (int i = 0; i < coordinate.length; i++) {
			index += coordinate[i]*strides[i];
		}
		return index + startIndex;
	}

	/**
	 * Specialized version of {@link #idx(int...)} for 1-D tensors.
	 *
	 * @param axis0 axis-0 of coordinate
	 * @return index in internal data array
	 */
	public int idx(int axis0 ) {
		return startIndex + axis0;
	}

	/**
	 * Specialized version of {@link #idx(int...)} for 2-D tensors.
	 *
	 * @param axis1 axis-1 of coordinate
	 * @param axis0 axis-0 of coordinate
	 * @return index in internal data array
	 */
	public int idx(int axis1, int axis0 ) {
		return startIndex + axis1*strides[0] + axis0;
	}

	/**
	 * Specialized version of {@link #idx(int...)} for 3-D tensors.
	 *
	 * @param axis2 axis-2 of coordinate
	 * @param axis1 axis-1 of coordinate
	 * @param axis0 axis-0 of coordinate
	 * @return index in internal data array
	 */
	public int idx(int axis2, int axis1 , int axis0 ) {
		return startIndex + axis2*strides[0] + axis1*strides[1] + axis0;
	}

	/**
	 * Specialized version of {@link #idx(int...)} for 4-D tensors.
	 *
	 * @param axis3 axis-3 of coordinate
	 * @param axis2 axis-2 of coordinate
	 * @param axis1 axis-1 of coordinate
	 * @param axis0 axis-0 of coordinate
	 * @return index in internal data array
	 */
	public int idx(int axis3, int axis2 , int axis1 , int axis0 ) {
		return startIndex + axis3*strides[0] + axis2*strides[1] + axis1*strides[2] + axis0;
	}

	/**
	 * Specialized version of {@link #idx(int...)} for 5-D tensors.
	 *
	 * @param axis4 axis-4 of coordinate
	 * @param axis3 axis-3 of coordinate
	 * @param axis2 axis-2 of coordinate
	 * @param axis1 axis-1 of coordinate
	 * @param axis0 axis-0 of coordinate
	 * @return index in internal data array
	 */
	public int idx(int axis4, int axis3 , int axis2 , int axis1 , int axis0 ) {
		return startIndex + axis4*strides[0] + axis3*strides[1] + axis2*strides[2] + axis1*strides[3] + axis0;
	}

	/**
	 * Computes how many indexes must be steped over to increment a value in each axis
	 */
	public void computeStrides() {
		if( strides.length != shape.length ) {
			strides = new int[ shape.length ];
		}
		int N = 1;
		for (int i = shape.length-1; i >= 0; i-- ) {
			strides[i] = N;
			N *= shape[i];
		}
	}

	/**
	 * Returns the stride at the specified dimension.  If negative it will start counting from the tail
	 */
	public int stride( int index ) {
		if( index < 0 ) {
			return strides[strides.length+index];
		} else {
			return strides[index];
		}
	}

	/**
	 * Copies the shape of this tensor into the provided int[]
	 * @param shape Where the shape is to be written to.  Must be the correct size
	 */
	public void copyShape( int shape[] ) {
		if( shape.length != this.shape.length )
			throw new IllegalArgumentException("Dimension of input shape and actual shape must be the same");

		for (int i = 0; i < shape.length; i++) {
			shape[i] = this.shape[i];
		}
	}

	/**
	 * Returns the length of a dimension/axis.  If a negative number is passed in it will
	 * return the distance relative to the end.  E.g. -1 = length-1, -2 = length-2
	 * @param dimension The dimension/axis
	 * @return length
	 */
	public int length(int dimension ) {
		if( dimension < 0 )
			return shape[ shape.length+dimension];
		else
			return shape[dimension];
	}

	/**
	 * Length of used elements in Tensor's data array.  Note that the actual data array can be larger
	 * @return length of used region in data array
	 */
	public int length() {
		if( shape.length == 0 )
			return 0;

		return shape[0]*strides[0];
	}

	/**
	 * Creates a tensor of the same type with the specified shape
	 * @param shape Shape of the new tensor
	 * @return New tensor with the specified shape
	 */
	public abstract T create( int ...shape );

	/**
	 * Creates a tensor of the same type and same shape as this one
	 * @return New tensor
	 */
	public T createLike() {
		return create(shape);
	}

	/**
	 * Creates a new coordinate for this tensor.
	 * @return coordinate/int array of appropriate length
	 */
	public int[] createCoor() {
		return new int[ getDimension() ];
	}

	/**
	 * Converts an array index into a coordinate
	 * @param index internal array index offset from startIndex
	 * @param storage (Optional) storage for coordinate.  If null a new instance is created
	 * @return coordinate
	 */
	public int[] indexToCoor( int index , int []storage ) {
		if( storage == null ) {
			storage = createCoor();
		}
		for (int i = 0; i < storage.length; i++) {
			storage[i] = index / strides[i];
			index -= storage[i]*strides[i];
		}
		return storage;
	}

	/**
	 * Turns 'this' tensor into a copy of the provided tensor.
	 * @param original Original tensor that's to be copied into this one.  Not modified.
	 */
	public void setTo( T original ) {
		reshape(original.getShape());
		System.arraycopy(original.getData(), original.startIndex, getData(), startIndex, length());
	}

	/**
	 * Sets all elements in the tensor to the value of zero
	 */
	public abstract void zero();

	/**
	 * Returns a copy of this tensor
	 */
	public T copy() {
		T out = createLike();
		out.setTo(this);
		return out;
	}

	/**
	 * Returns true if it is a sub-tensor.
	 *
	 * @return true if sub-tensor or false if not
	 */
	public boolean isSub() {
		return subtensor;
	}

	/**
	 * Creates a subtensor from this tensor.
	 * @param startIndex The start index in this tensor's internal data array
	 * @param shape Shape of the output tensor
	 * @return The new sub-tensor.
	 *
	 */
	public T subtensor(int startIndex, int[] shape) {
		T out = create();
		out.setData(getData());
		out.startIndex = startIndex;
		out.shape = shape;
		out.subtensor = true;
		out.computeStrides();
		return out;
	}
}
