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

package deepboof.impl.backward.standard;

import deepboof.PaddingType;
import deepboof.backward.CheckDerivativePadding;
import deepboof.forward.ConfigPadding;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
public class TestBackwards_DClippedPadding2D_F64
		extends CheckDerivativePadding<Tensor_F64,DClippedPadding2D_F64> {

	@Override
	public DClippedPadding2D_F64 createBackwards() {

		ConfigPadding config = new ConfigPadding();
		config.x0 = 1;
		config.x1 = 2;
		config.y0 = 3;
		config.y1 = 4;
		config.type = PaddingType.CLIPPED;

		return new DClippedPadding2D_F64(config);
	}

	@Override
	protected void applyPadding(Tensor_F64 input, Tensor_F64 output) {
		alg.setInput(input);

		for (int batch = 0; batch < input.length(0); batch++) {
			for (int channel = 0; channel < input.length(1); channel++) {
				for (int row = 0; row < output.length(2); row++) {
					for (int col = 0; col < output.length(3); col++) {
						output.d[output.idx(batch,channel,row,col)] = alg.get(batch,channel,row,col);
					}
				}
			}
		}
	}
}