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
 * Tensor Interface
 *
 * @author Peter Abeles
 */
public interface ITensor {
	/**
	 * Returns the internal array that specifies the tensor's shape. DO NOT MODIFIY
	 *
	 * @return int[] array of the shape
	 */
	int[] getShape();

	/**
	 * Checks to see if the supplied shape is the same as the tensor's shape
	 * @param shape int[] which specifies the shape ofa tensor
	 * @return true if the same or false if not
	 */
	boolean isShape(int ...shape );

	/**
	 * Returns the length of a dimension/axis.  If a negative number is passed in it will
	 * return the distance relative to the end.  E.g. -1 = length-1, -2 = length-2
	 * @param dimension The dimension/axis
	 * @return length
	 */
	int length(int dimension );

	/**
	 * Creates a new coordinate for this tensor.
	 * @return coordinate/int array of appropriate length
	 */
	int[] createCoor();

	/**
	 * Returns the number of dimensions in the tensor
	 * @return number of dimensions
	 */
	int getDimension();

	/**
	 * {@link Class} of primitive data type used to store tensor
	 * @return Internal data type class.
	 */
	Class getDataType();
}
