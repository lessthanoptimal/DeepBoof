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

import deepboof.Function;
import deepboof.Tensor;

/**
 * The sigmoid is defined as:
 * <p>
 * &sigma;(x) = 1/(1 + e<sup>-x</sup>)
 * </p>
 * It has a range from 0 to 1.  It converges towards one for positive numbers and zero for negative numbers.
 *
 * @author Peter Abeles
 */
public interface ActivationSigmoid<T extends Tensor> extends Function<T> {

}
