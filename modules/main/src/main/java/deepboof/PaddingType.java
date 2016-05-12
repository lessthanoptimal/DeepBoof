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
 * Specifies the type of padding applied to a spacial function.
 *
 * @author Peter Abeles
 */
public enum PaddingType {
	/**
	 * Input is padded with zero valued elements
	 */
	ZERO,
	/**
	 * Input is padded with the most negative possible number
	 */
	MAX_NEGATIVE,
	/**
	 * Input is padded with the most positive possible number
	 */
	MAX_POSITIVE,
	/**
	 * Input is padded with values which extend the nearest element
	 */
	EXTEND,
	/**
	 * The kernel is cropped and reweighted such that it does not extend outside the image.
	 */
	KERNEL_CROP
}
