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

package deepboof.impl.forward.standard;

import deepboof.DeepBoofConstants;
import deepboof.PaddingType;
import deepboof.forward.ConfigPadding;
import deepboof.misc.TensorFactory_F32;
import deepboof.tensors.Tensor_F32;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * @author Peter Abeles
 */
public class TestConstantPadding2D_F32 {
	Random rand = new Random(234);

		@Test
	public void isClipped() {
		ConstantPadding2D_F32 alg = new ConstantPadding2D_F32(new ConfigPadding());

		assertFalse(alg.isClipped());
	}

	/**
	 * Determine the value from the padding type
	 */
	@Test
	public void fromConfig() {
		basic(PaddingType.ZERO,0);
		basic(PaddingType.MAX_NEGATIVE,-Float.MAX_VALUE);
		basic(PaddingType.MAX_POSITIVE, Float.MAX_VALUE);
	}

	private void basic(PaddingType type, float expected) {
		ConfigPadding config = new ConfigPadding();
		config.x0=1;
		config.x1=2;
		config.y0=3;
		config.y1=4;
		config.type = type;

		ConstantPadding2D_F32 alg = new ConstantPadding2D_F32(config);

		Tensor_F32 tensor = TensorFactory_F32.randomMM(rand,false,-1,1,    4,3,6,2);

		alg.setInput(tensor);

		// inside the physical tensor
		assertEquals(tensor.get(1,2,1,0),alg.get(1,2,4,1), DeepBoofConstants.TEST_TOL_F32);

		// in the border
		assertEquals(expected,alg.get(1,2,0,0), DeepBoofConstants.TEST_TOL_F32);
		assertEquals(expected,alg.get(1,2,9,0), DeepBoofConstants.TEST_TOL_F32);
		assertEquals(expected,alg.get(1,2,0,3), DeepBoofConstants.TEST_TOL_F32);
	}
}