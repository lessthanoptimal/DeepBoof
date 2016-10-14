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

import deepboof.DeepBoofConstants;
import deepboof.DummyFunction;
import deepboof.tensors.Tensor_F64;
import org.ddogleg.struct.Tuple2;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class TestForwardSequence {
	/**
	 * Simple test with a linear function that has no combine functions
	 */
	@Test
	public void process_linear_nocombine() {
		int numBatch = 1;
		for (int length = 1; length <= 5; length++) {
			List<Node> sequence = createLine(length);

			FunctionSequence alg = new FunctionSequence(sequence, Tensor_F64.class);
			alg.initialize(new int[]{1});

			// actual input tensors are always (mini-batch, .... etc.) hence the extra 1 dimension
			Tensor_F64 input = new Tensor_F64(numBatch,1);
			Tensor_F64 output = new Tensor_F64(numBatch,length);

			input.d[0] = 0;

			alg.process(input,output);

			for (int i = 0; i < length; i++) {
				assertEquals(length, output.get(0,i), DeepBoofConstants.TEST_TOL_F64);
			}

		}
	}

	@Test
	public void setParameters() {
		List<Node> sequence = createLine(10);
		Map<String,List<Tensor_F64>> parameters = new HashMap<>();
		List<List<Tensor_F64>> expected = new ArrayList<>();
		for (Node node : sequence) {
			List<Tensor_F64> p = new ArrayList<>();
			expected.add(p);
			parameters.put(node.name, p);
		}

		FunctionSequence alg = new FunctionSequence(sequence, Tensor_F64.class);

		alg.setParameters(parameters);

		for (int i = 0; i < sequence.size(); i++) {
			List<Tensor_F64> found =  ((HelperFunction)((Node)alg.sequence.get(i)).function).parameters;
			assertTrue(found == expected.get(i));
		}
	}

	@Test
	public void initialize() {
		for (int length = 1; length <= 5; length++) {
			List<Node> sequence = createLine(length);

			FunctionSequence alg = new FunctionSequence(sequence, Tensor_F64.class);

			alg.initialize(new int[]{10});

			for (int i = 0; i < length; i++) {
				HelperFunction f = (HelperFunction) sequence.get(i).function;
				assertEquals(10 + i + 1, f.output[0]);

				Tuple2 functions = (Tuple2) alg.outputStorage.get("" + i);

				// should predeclare an empty tensor
				assertTrue(functions.data0 != null);
			}
		}
	}

	private static List<Node> createLine( int length ) {
		List<Node> out = new ArrayList<>();

		for (int i = 0; i < length; i++) {
			out.add(create(""+i));
		}

		for (int j = 1; j < length; j++) {
			out.get(j).sources.add( new InputAddress(""+(j-1)));
		}

		return out;
	}

	private static Node create( String name ) {
		Node n = new Node();
		n.name = name;
		n.function = new HelperFunction();
		return n;
	}

	private static class HelperFunction extends DummyFunction<Tensor_F64> {
		int output[] = new int[1];

		List<Tensor_F64> parameters;

		@Override
		public void initialize(int... shapeInput) {
			output[0] = shapeInput[0]+1;
		}

		@Override
		public void setParameters(List<Tensor_F64> parameters) {
			this.parameters = parameters;
		}

		@Override
		public void forward(Tensor_F64 input, Tensor_F64 output) {
			double a = input.d[0];

			for (int i = 0; i < output.length(); i++) {
				output.d[output.idx(i)] = a+1;
			}
		}

		@Override
		public int[] getOutputShape() {
			return output;
		}
	}
}
