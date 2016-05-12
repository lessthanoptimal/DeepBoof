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

package deepboof.tensors;

import deepboof.VTensor;

/**
 * Virtual tensor for 64bit float types.
 *
 * @author Peter Abeles
 */
public interface VTensor_F64 extends VTensor {

	/**
	 * Returns the element's value at the specified coordinate.
	 *
	 * @param coor coordinate at ordered from axis-(N-1) to axis-0
	 * @return element value
	 */
	double get( int ...coor );

	/**
	 * Returns the element's value at the specified coordinate.  Only works with 1-DOF
	 * tensors.
	 *
	 * @param axis0 coordinate at axis-0
	 * @return element value
	 */
	double get( int axis0 );

	/**
	 * Returns the element's value at the specified coordinate.  Only works with 2-DOF
	 * tensors.
	 *
	 * @param axis1 coordinate at axis-1
	 * @param axis0 coordinate at axis-0
	 * @return element value
	 */
	double get( int axis1 , int axis0 );

	/**
	 * Returns the element's value at the specified coordinate.  Only works with 3-DOF
	 * tensors.
	 *
	 * @param axis2 coordinate at axis-2
	 * @param axis1 coordinate at axis-1
	 * @param axis0 coordinate at axis-0
	 * @return element value
	 */
	double get( int axis2 , int axis1 , int axis0 );

	/**
	 * Returns the element's value at the specified coordinate.  Only works with 4-DOF
	 * tensors.
	 *
	 * @param axis3 coordinate at axis-3
	 * @param axis2 coordinate at axis-2
	 * @param axis1 coordinate at axis-1
	 * @param axis0 coordinate at axis-0
	 * @return element value
	 */
	double get( int axis3 , int axis2 , int axis1 , int axis0 );

	/**
	 * Returns the element's value at the specified coordinate.  Only works with 5-DOF
	 * tensors.
	 *
	 * @param axis4 coordinate at axis-4
	 * @param axis3 coordinate at axis-3
	 * @param axis2 coordinate at axis-2
	 * @param axis1 coordinate at axis-1
	 * @param axis0 coordinate at axis-0
	 * @return element value
	 */
	double get( int axis4, int axis3 , int axis2 , int axis1 , int axis0 );
}
