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

package deepboof.io.torch7;

import deepboof.Function;
import deepboof.Tensor;
import deepboof.graph.FunctionSequence;
import deepboof.graph.Node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Storage for a saved DeepBoof a sequence of functions and their learned parameters
 *
 * @author Peter Abeles
 */
public class SequenceAndParameters< T extends Tensor<T>,F extends Function<T>> {
	/**
	 * Set of functions with auxillary information, e.g. name
	 */
	public List<Node<T,F>> sequence = new ArrayList<>();
	/**
	 * Map of learned parameters.  The key is the function's name
	 */
	public Map<String,List<T>> parameters = new HashMap<>();
	public Class<T> type;

	public FunctionSequence<T,F> createForward(int ...shape) {
		FunctionSequence<T,F> network = new FunctionSequence<>(sequence,type);
		network.initialize(shape);
		network.setParameters(parameters);
		return network;
	}
}
