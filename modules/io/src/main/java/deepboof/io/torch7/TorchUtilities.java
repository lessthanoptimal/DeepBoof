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
import deepboof.graph.FunctionSequence;
import deepboof.tensors.Tensor_F32;

import java.io.File;
import java.io.IOException;

/**
 * Application which runs through the user specified model or all the models to see if the input produces
 * the expected output.  Expected output was generated inside of torch.
 *
 *
 * @author Peter Abeles
 */
public class TorchUtilities {

	public static boolean validateNetwork( File directory , boolean exitOnFail ) throws IOException {
		ParseBinaryTorch7 parser = new ParseBinaryTorch7();

		SequenceAndParameters<Tensor_F32, Function<Tensor_F32>> sequence =
				parser.parseIntoBoof(new File(directory,"model.net"));

		FunctionSequence<Tensor_F32,Function<Tensor_F32>> network = sequence.createForward(3,32,32);

		Tensor_F32 input = parser.parseIntoBoof(new File(directory,"test_input.t7"));
		Tensor_F32 expected = parser.parseIntoBoof(new File(directory,"test_output.t7"));
		Tensor_F32 found = expected.createLike();

		network.process(input,found);

		for (int i = 0; i < expected.length(); i++) {
			double error = Math.abs(expected.d[i] - found.d[i]);
			if( error > 1e-3 ) {
				if( exitOnFail ) {
					System.err.println("network test failed at "+i+"  error = "+error);
					System.exit(1);
				}
				return false;
			}
		}
		return true;
	}

	public static void main(String[] args) throws IOException {
		File directory = new File("data/torch_models/likevgg_cifar10");

		System.out.println("Loading and evaluating...");

		validateNetwork(directory,true);

		System.out.println("Passed!");
	}
}
