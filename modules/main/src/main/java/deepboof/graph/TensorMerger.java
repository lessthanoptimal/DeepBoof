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

package deepboof.graph;

import deepboof.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * Merged multiple input tensors into a single output which can be processed by a {@link Function}.
 *
 * @author Peter Abeles
 */
public interface TensorMerger<T extends Tensor<T>> {

	void initialize( List<int[]> inputShapes );

	void combine(List<T> inputs , T output );

	int[] getOutputShape();
}
