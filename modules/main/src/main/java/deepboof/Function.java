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

import java.util.List;

/**
 * <p>High level interface for functions in an Artificial Neural Network.  This interface only defines the
 * the operations in the forwards pass.  When learning a network the gradient is typically needed and
 * those additional operations can be found in {@link DFunction}, which extends this interface.</p>
 *
 * <p>Forwards only implementations potentially have a lower memory foot print, faster specialized
 * implementations, more simplistic implementations.</p>
 *
 * @author Peter Abeles
 */
public interface Function< T extends Tensor > {

	/**
	 * Initializes internal data structures given the shape of the input tensor, minus the stacked input
	 * dimension.
	 *
	 * For example, an input tensor of shape (B,C,D) might be passed into initialize, while the actual input
	 * is (N,B,C,D).  N is the number of stacked inputs and is allowed to vary after initialization.
	 *
	 * @throws IllegalArgumentException If input tensor shapes are not valid
	 * @param shapeInput Shape of the input tensor
	 */
	void initialize( int... shapeInput );

	/**
	 * <p>Specifies learnable function parameters, e.g. weights for linear functions.  This function only
	 * needs to be called once each time a parameter has been modified.  Must be called before {@link #forward}.</p>
	 *
	 * NOTE: Reference to the parameters may be saved internally and the tensors should not be modified externally.
	 *
	 * @param parameters Tensors containing parameters which are optimized.  Not modified.
	 */
	void setParameters( List<T> parameters );

	/**
	 * If the parameters have been set, then this returns the list of parameters.  Otherwise null is returned.
	 *
	 * @return List of parameters or null if they have not been set yet
	 */
	List<T> getParameters();

	/**
	 * Performs forward pass of the function on the provided inputs.
	 *
	 * <pre>
	 * Input tensor shape = (N,variable ... )
	 * - N is the mini-batch size
	 * - Other dimensions are implementation specific.
	 * </pre>
	 *
	 * @param input Input to the function.
	 * @param output Output tensor. Modified.
	 */
	void forward( T input , T output );

	/**
	 * Returns the shape of input tensors, without the mini-batch dimension.
	 * Only valid after {@link #initialize} has been called.
	 *
	 * @return Expected shapes of input tensors.  This data structure may be recycled and is modified on the next
	 * call to {@link #initialize}.
	 */
	List<int[]> getParameterShapes();

	/**
	 * Returns the output tensor's shape, without the mini-batch dimension.
	 * Only valid after {@link #initialize} has been called.
	 *
	 * @return Expected shape of output tensor.  This data structure may be recycled and is modified on the next
	 * call to {@link #initialize}.
	 */
	int[] getOutputShape();

	/**
	 * Returns the type of tensor it can process
	 *
	 * @return Type of tensor
	 */
	Class<T> getTensorType();
}
