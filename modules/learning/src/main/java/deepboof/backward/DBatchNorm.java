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

package deepboof.backward;

import deepboof.DFunction;
import deepboof.Tensor;
import deepboof.forward.BatchNorm;

/**
 * Implementation of {@link deepboof.forward.BatchNorm batch normalization} for training networks.
 * The mean and standard deviation is always computed on the forwards pass.  Unlike the forward only
 * implementation the only parameters (which are optional) are gamma and beta.
 *
 * @author Peter Abeles
 */
public interface DBatchNorm<T extends Tensor<T>> extends BatchNorm, DFunction<T> {
    /**
     * Returns the most recently computed mean for each variable in the tensor.
     *
     * @param output Storage for mean tensor. Is reshaped. If null a new instance will be declared
     */
    T getMean( T output );

    /**
     * Returns the most recently computed variance for each variable.  This will be the actual variance not something that has been
     * adjusted by adding EPS to it.
     *
     * @param output Storage for variance tensor. Is reshaped. If null a new instance will be declared
     */
    T getVariance( T output );
}
