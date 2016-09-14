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

import deepboof.backward.DSpatialPadding2D;
import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.forward.ConfigPadding;
import deepboof.impl.forward.standard.ConstantPadding2D_F64;
import deepboof.misc.TensorOps_F64;
import deepboof.tensors.Tensor_F64;

/**
 * Backwards implementation of {@link ConstantPadding2D_F64}.
 *
 * @author Peter Abeles
 */
public class DConstantPadding2D_F64 extends ConstantPadding2D_F64
		implements DSpatialPadding2D<Tensor_F64>, DSpatialPadding2D_F64
{
	public DConstantPadding2D_F64(ConfigPadding config) {
		super(config);
	}

	@Override
	public void backwardsChannel(Tensor_F64 gradientPadded, int batch, int channel,
								 Tensor_F64 gradientInput)
	{
		checkBackwardsShapeChannel(gradientPadded,gradientInput);

		// Padded gradient is a 2D tensor
		int indexSrc = gradientPadded.idx(ROW0,COL0);
		int strideSrc = gradientPadded.length(1);

		// gradient input is a full 4D spatial tensor
		int indexDst = gradientInput.idx(batch,channel,0,0);
		int strideDst = gradientInput.length(3);

		// copy only the inner portion of the padded gradient into the input gradient,  The border is all zero
		TensorOps_F64.insertSubChannel(gradientPadded,indexSrc,strideSrc,gradientInput,indexDst,strideDst,
				gradientInput.length(2),gradientInput.length(3));
	}

	@Override
	public void backwardsImage(Tensor_F64 gradientPadded, int batch, Tensor_F64 gradientInput) {
		checkBackwardsShapeImage(gradientPadded,gradientInput);

		final int numChannels = gradientPadded.length(0);
		final int imgHeight = gradientInput.length(2);
		final int imgWidth = gradientInput.length(3);

		final int strideSrc = gradientPadded.length(2);
		final int strideDst = gradientInput.length(3);

		for (int channel = 0; channel < numChannels; channel++) {
			// Padded gradient is a 2D tensor
			int indexSrc = gradientPadded.idx(channel,ROW0,COL0);

			// gradient input is a full 4D spatial tensor
			int indexDst = gradientInput.idx(batch,channel,0,0);

			// copy only the inner portion of the padded gradient into the input gradient,  The border is all zero
			TensorOps_F64.insertSubChannel(
					gradientPadded,indexSrc,strideSrc,
					gradientInput,indexDst,strideDst,
					imgHeight,imgWidth);
		}

	}
}
