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
 * Various constants used throughout DeepBoof
 *
 * @author Peter Abeles
 */
public class DeepBoofConstants {
	/**
	 * Standard tolerance for unit tests in F64
	 */
	public static double TEST_TOL_F64 = 1e-8;
	public static double TEST_TOL_A_F64 = 1e-6;
	public static double TEST_TOL_B_F64 = 1e-4;

	/**
	 * Standard tolerance for unit tests in F32
	 */
	public static float TEST_TOL_F32 = 1e-4F;
	public static float TEST_TOL_A_F32 = 1e-3F;
	public static float TEST_TOL_B_F32 = 1e-2F;

}
