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

import deepboof.Function;
import deepboof.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Node in a network graph which describes the network's processing sequence.
 *
 * If there are multiple sources a {@link TensorMerger} must be provided.  A {@link Function function} can
 * only process one input.
 *
 * @author Peter Abeles
 */
public class Node<T extends Tensor<T>, F extends Function<T>> {
	/**
	 * Specifies locations of inputs to this function
	 */
	public List<InputAddress> sources = new ArrayList<>();
	/**
	 * Unique identifier for this function
	 */
	public String name;
	/**
	 * Operation for combining multiple input sources together
	 */
	public TensorMerger<T> combine;
	/**
	 * Function for this node
	 */
	public F function;
}
