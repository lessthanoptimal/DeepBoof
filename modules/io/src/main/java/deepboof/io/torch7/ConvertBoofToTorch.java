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

import deepboof.Tensor;
import deepboof.io.torch7.struct.*;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;
import deepboof.tensors.Tensor_U8;

/**
 * @author Peter Abeles
 */
public class ConvertBoofToTorch {
	/**
	 * Converts a DeepBoof tensor into a Torch tensor
	 */
	public static TorchTensor convert( Tensor input ) {
		TorchTensor torch = new TorchTensor();
		torch.version = 1;
		torch.shape = input.shape;

		if( input.isSub() )
			throw new IllegalArgumentException("Subtensors not yet supported");

		if( input instanceof Tensor_F64) {
			Tensor_F64 t = (Tensor_F64)input;
			TorchDoubleStorage storage = new TorchDoubleStorage(0);
			storage.data = t.d;
			storage.version = 1;

			torch.storage = storage;
			torch.torchName = "torch.DoubleTensor";
		} else if( input instanceof Tensor_F32 ) {
			Tensor_F32 t = (Tensor_F32)input;
			TorchFloatStorage storage = new TorchFloatStorage(0);
			storage.data = t.d;
			storage.version = 1;

			torch.storage = storage;
			torch.torchName = "torch.FloatTensor";
		} else if( input instanceof Tensor_U8) {
			Tensor_U8 t = (Tensor_U8)input;
			TorchByteStorage storage = new TorchByteStorage(0);
			storage.data = t.d;
			storage.version = 1;

			torch.storage = storage;
			torch.torchName = "torch.ByteTensor";
		} else {
			throw new RuntimeException("Add support for "+input.getClass().getSimpleName());
		}

		return torch;
	}

	public static TorchBoolean convert( boolean value ) {
		TorchBoolean output = new TorchBoolean();
		output.value = value;
		return output;
	}

	public static TorchNumber convert( double number ) {
		TorchNumber output = new TorchNumber();
		output.value = number;
		return output;
	}

	public static TorchString convert( String message ) {
		TorchString output = new TorchString();
		output.message = message;
		return output;
	}
}
