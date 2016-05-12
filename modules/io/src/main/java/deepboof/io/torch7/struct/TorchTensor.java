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

package deepboof.io.torch7.struct;

/**
 * @author Peter Abeles
 */
public class TorchTensor extends TorchReferenceable {
	public int shape[];
	public int startIndex;
	public TorchStorage storage;

	public String toString() {
		String out = torchName;
		out += "{ ";
		if( shape != null ) {
			out += "shape=(";
			for (int i = 0; i < shape.length; i++) {
				out += shape[i];
				if( i < shape.length-1)
					out += " , ";
			}
			out += ") ";
			out += " start="+startIndex+" ";
		}
		out += "}";
		return out;
	}

	public int length() {
		int total = 1;
		for (int i = 0; i < shape.length; i++) {
			total *= shape[i];
		}
		return total;
	}
}
