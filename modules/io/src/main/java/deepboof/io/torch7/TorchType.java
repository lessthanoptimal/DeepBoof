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

/**
 * Enumerated type for the different types of Torch data structures. Values derived from Torch source code.
 *
 * @author Peter Abeles
 */
public enum TorchType {

	NIL(0),
	NUMBER(1),
	STRING(2),
	TABLE(3),
	TORCH(4),
	BOOLEAN(5),
	FUNCTION(6),
	RECUR_FUNCTION(8),
	LEGACY_RECUR_FUNCTION(7);

	TorchType( int value  ) {
		this.value = value;
	}

	public static TorchType valueToType( int value ) {
		switch( value ) {
			case 0: return NIL;
			case 1: return NUMBER;
			case 2: return STRING;
			case 3: return TABLE;
			case 4: return TORCH;
			case 5: return BOOLEAN;
			case 6: return FUNCTION;
			case 7: return LEGACY_RECUR_FUNCTION;
			case 8: return RECUR_FUNCTION;
			default:
				throw new RuntimeException(String.format("Unknown type.  Value = 0x%08x",value));
		}
	}

	int value;

	public int getValue() {
		return value;
	}
}
