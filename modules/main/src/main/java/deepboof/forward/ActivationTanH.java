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
 * The hyperbolic tangent (tanh) is defined as:
 * <pre>
 * tanh(x) = sinh(x)/cosh(x) = 2*&sigma;(2*x) - 1
 * </pre>
 * where &sigma;(x) is the sigmoid function, sinh and cosh are hyperbolic sine and cosine functions.  It has
 * a range of -1 to 1.  Converges towards 1 for positive numbers and -1 for negative numbers.
 *
 * @author Peter Abeles
 */
public interface ActivationTanH<T extends Tensor> extends Function<T> {
}
