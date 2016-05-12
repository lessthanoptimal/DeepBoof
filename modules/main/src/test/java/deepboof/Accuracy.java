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
 * Selects level of accuracy in a data type agnostic way
 *
 * @author Peter Abeles
 */
public enum Accuracy {
	/**
	 * Standard accuracy.  F64 = {@link DeepBoofConstants#TEST_TOL_F64}
	 */
	STANDARD,
	/**
	 * Relaxed accuracy.  F64 = {@link DeepBoofConstants#TEST_TOL_A_F64}
	 */
	RELAXED_A,
	/**
	 * Relaxed accuracy.  F64 = {@link DeepBoofConstants#TEST_TOL_B_F64}
	 */
	RElAXED_B;

	public double value( Class type ) {
		if( type == double.class ) {
			switch (this) {
				case STANDARD:
					return DeepBoofConstants.TEST_TOL_F64;
				case RELAXED_A:
					return DeepBoofConstants.TEST_TOL_A_F64;
				case RElAXED_B:
					return DeepBoofConstants.TEST_TOL_B_F64;
			}
		} else if( type == float.class ) {
			switch (this) {
				case STANDARD:
					return DeepBoofConstants.TEST_TOL_F32;
				case RELAXED_A:
					return DeepBoofConstants.TEST_TOL_A_F32;
				case RElAXED_B:
					return DeepBoofConstants.TEST_TOL_B_F32;
			}
		}
		throw new IllegalArgumentException("Unknown something");
	}
}
