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

package deepboof.forward;

import deepboof.DeepBoofConstants;
import deepboof.misc.TensorFactory_F32;
import deepboof.tensors.Tensor_F32;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public class TestSpatialPadding2D_F32 {

	Random rand = new Random(234);

	@Test
	public void get_four_all() {
		ConfigPadding config = new ConfigPadding();
		config.x0=1;
		config.x1=2;
		config.y0=3;
		config.y1=4;

		Helper helper = new Helper(config);

		Tensor_F32 tensor = TensorFactory_F32.randomMM(rand,false,-1,1,    4,3,6,2);

		helper.setInput(tensor);

		// inside the physical tensor
		assertEquals(tensor.get(1,2,1,0),helper.get(1,2,4,1), DeepBoofConstants.TEST_TOL_F32);
		assertEquals(tensor.get(1,2,1,0),helper.get(new int[]{1,2,4,1}), DeepBoofConstants.TEST_TOL_F32);

		// in the border
		assertEquals(1.0f+2.0f,helper.get(1,2,0,0), DeepBoofConstants.TEST_TOL_F32);
		assertEquals(1.0f+2.0f,helper.get(new int[]{1,2,0,0}), DeepBoofConstants.TEST_TOL_F32);
	}

	public static class Helper extends SpatialPadding2D_F32 {

		public Helper(ConfigPadding config) {
			super(config);
		}

		@Override
		public float borderGet(int minibatch, int channel, int row, int col) {
			return minibatch+channel+row+col;
		}

		@Override
		public int getClippingOffsetRow(int paddedRow) {
			return 0;
		}

		@Override
		public int getClippingOffsetCol(int paddedCol) {
			return 0;
		}

		@Override
		public boolean isClipped() {
			return false;
		}

		@Override
		public Class<Tensor_F32> getTensorType() {
			return Tensor_F32.class;
		}
	}
}