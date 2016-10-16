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

package deepboof.forward;

import deepboof.Tensor;
import deepboof.VTensor;

/**
 * <p>Interface for all virtual 2D spatial padding implementation.  Virtual padding contains a reference
 * to the original input tensor which is going to be padded and on the fly will generate the values for
 * elements which are not explicitly contained in the input tensor.  This can reduce memory consumption and is
 * more simplistic to implement for more complex padding methods.</p>
 *
 * Clipped padding is a special case.  In this situation only pixels contained inside the original image should
 * be processed.
 *
 * @author Peter Abeles
 */
public interface SpatialPadding2D<T extends Tensor<T>> extends VTensor {
	/**
	 * Spatial tensor that padding is being added around
	 * @param input The input tensor
	 */
	void setInput(T input);

	/**
	 * Returns how far away the row is from the clipping border.  0 if it is inside the image. Positive is below
	 * the lower extent and negative if above upper extent.
	 *
	 * @param paddedRow Row in padded coordinates
	 * @return offset
	 */
	int getClippingOffsetRow( int paddedRow );

	/**
	 * Returns how far away the column is from the clipping border.  0 if it is inside the image. Positive is below
	 * the lower extent and negative if above upper extent.
	 *
	 * @param paddedCol Column in padded coordinates
	 * @return offset
	 */
	int getClippingOffsetCol( int paddedCol );

	/**
	 * Returns the lower-extent padding along the tensor's rows.
	 * @return padding
	 */
	int getPaddingRow0();

	/**
	 * Returns the lower-extent padding along the tensor's columns.
	 * @return padding
	 */
	int getPaddingCol0();

		/**
	 * Returns the upper-extent padding along the tensor's rows.
	 * @return padding
	 */
	int getPaddingRow1();

	/**
	 * Returns the upper-extent padding along the tensor's columns.
	 * @return padding
	 */
	int getPaddingCol1();

	/**
	 * Returns what the tensor's shape will be when given can input tensor with
	 * the spcified shape.
	 * @param inputShape Input spatial tensor.  3-DOF with no mini-batch or 4-DOF with mini-batch
	 * @return Tensor's shape
	 */
	int[] shapeGivenInput( int ...inputShape );

	/**
	 * Returns true if this is a clipped border or false of it is not.
	 * @return if clipped or not
	 */
	boolean isClipped();

	/**
	 * Returns the type of input tensor it can process
	 *
	 * @return Type of tensor
	 */
	Class<T> getTensorType();
}
