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

import deepboof.BaseTensor;
import deepboof.Tensor;
import deepboof.forward.ConfigPadding;
import deepboof.forward.SpatialPadding2D;

/**
 * Abstract class fo all virtual 2D spatial padding implementation.  Virtual padding contains a reference
 * to the original input tensor which is going to be padded and on the fly will generate the values for
 * elements which are not explicitly contained in the input tensor.  This can reduce memory consumption and is
 * more simplistic to implement for more complex padding methods.
 *
 * @author Peter Abeles
 */
public abstract class BaseSpatialPadding2D<T extends Tensor<T>>
		extends BaseTensor implements SpatialPadding2D<T>
{
	// description of how to apply the padding
	protected ConfigPadding config;

	// input image
	protected T input;

	// boundary of input tensor in the virtual padded image
	protected int ROW0,ROW1,COL0,COL1;

	public BaseSpatialPadding2D(ConfigPadding config) {
		this.config = config;
	}

	/**
	 * {@inheritDoc}
	 */
	public void setInput(T input) {
		if( input.getDimension() != 4 )
			throw new IllegalArgumentException("Expected 4-DOF spatial tensor");

		this.input = input;

		int rows = input.length(2);
		int cols = input.length(3);

		COL0 = config.x0;
		ROW0 = config.y0;
		COL1 = cols+config.x0;
		ROW1 = rows+config.y0;

		this.shape = shapeGivenInput(input.shape);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getPaddingRow0() {
		return config.y0;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getPaddingCol0() {
		return config.x0;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getPaddingRow1() {
		return config.y1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getPaddingCol1() {
		return config.x1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int[] shapeGivenInput( int ...inputShape ) {
		if( inputShape.length == 3)
			return new int[]{
					inputShape[0],
					inputShape[1]+config.y0 +config.y1,
					inputShape[2]+config.x0 +config.x1,
			};
		else if( inputShape.length == 4 ) {
			return new int[]{
					inputShape[0],
					inputShape[1],
					inputShape[2]+config.y0 +config.y1,
					inputShape[3]+config.x0 +config.x1,
			};
		} else {
			throw new IllegalArgumentException("Spatial tensor with 3 or 4 dof expected");
		}
	}

	/**
	 * Sanity checks the input for backwards images
	 * @param padded Input padded single channel image from forward layer
	 * @param original Output backwards spatial tensor
	 */
	public <T extends Tensor<T>>
	void checkBackwardsShapeChannel( Tensor<T> padded , Tensor<T> original ) {
		if( padded.getDimension() != 2 )
			throw new IllegalArgumentException("Padded image expected to be a 2D spatial image, i.e. 2 channels");
		if( original.getDimension() != 4 )
			throw new IllegalArgumentException("Original image expected to be a 4D spatial image, i.e. 4 channels");

		if( padded.length(0) != original.length(2)+config.y0+config.y1 ) {
			throw new IllegalArgumentException(
					"Image heights do not match.  "+padded.length(0)+" != "+original.length(2)+config.y0+config.y1);
		}
		if( padded.length(1) != original.length(3)+config.x0+config.x1) {
			throw new IllegalArgumentException(
					"Image widths do not match.  "+padded.length(1)+" != "+original.length(3)+config.x0+config.x1);
		}
	}

	/**
	 * Sanity checks the input for backwards images
	 * @param padded Input padded image from forward layer
	 * @param original Output backwards spatial tensor
	 */
	public <T extends Tensor<T>>
	void checkBackwardsShapeImage( Tensor<T> padded , Tensor<T> original ) {
		if( padded.getDimension() != 3 )
			throw new IllegalArgumentException("Padded image expected to be a 3D spatial image, i.e. 3 channels");
		if( original.getDimension() != 4 )
			throw new IllegalArgumentException("Original image expected to be a 4D spatial image, i.e. 4 channels");

		if( padded.length(0) != original.length(1) ) {
			throw new IllegalArgumentException(
					"Image channels do not match.  "+padded.length(0)+" != "+original.length(1));
		}
		if( padded.length(1) != original.length(2)+config.y0+config.y1 ) {
			throw new IllegalArgumentException(
					"Image heights do not match.  "+padded.length(1)+" != "+original.length(2)+config.y0+config.y1);
		}
		if( padded.length(2) != original.length(3)+config.x0+config.x1 ) {
			throw new IllegalArgumentException(
					"Image widths do not match.  "+padded.length(2)+" != "+original.length(3)+config.x0+config.x1);
		}
	}
}
