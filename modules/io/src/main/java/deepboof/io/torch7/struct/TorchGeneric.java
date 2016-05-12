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

import deepboof.io.torch7.ConvertTorchToBoofForward;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.Tensor_U8;

import java.util.HashMap;
import java.util.Map;

/**
 * Generic torch data structure.  Not designed for any specific data structure.
 *
 * @author Peter Abeles
 */
public class TorchGeneric extends TorchReferenceable {

	public Map<Object,TorchObject> map = new HashMap<>();

	public double getNumber( String key ) {
		TorchNumber number = (TorchNumber)map.get(key);
		return number.value;
	}

	public <T>T get(String key) {
		return (T)map.get(key);
	}

	public Tensor_U8 getTensorU8(String key ) {
		return (Tensor_U8)ConvertTorchToBoofForward.convert(map.get(key));
	}

	public Tensor_F32 getTensorF32(String key ) {
		return (Tensor_F32)ConvertTorchToBoofForward.convert(map.get(key));
	}

	public Tensor_F64 getTensorF64(String key ) {
		return (Tensor_F64)ConvertTorchToBoofForward.convert(map.get(key));
	}
}
