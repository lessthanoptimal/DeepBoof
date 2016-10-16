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
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
public class TestClippedPadding2D_F64 {

	Random rand = new Random(234);

	@Test
	public void isClipped() {
		ClippedPadding2D_F64 alg = createAlg();

		assertTrue(alg.isClipped());
	}

	@Test
	public void getClippingOffsetRow() {
		ClippedPadding2D_F64 alg = createAlg();

		Tensor_F64 tensor = TensorFactory_F64.randomMM(rand,false,-1,1,    4,3,6,2);
		alg.setInput(tensor);

		assertEquals(3,alg.getClippingOffsetRow(0));
		assertEquals(2,alg.getClippingOffsetRow(1));
		assertEquals(0,alg.getClippingOffsetRow(3));
		assertEquals(0,alg.getClippingOffsetRow(9));
		assertEquals(-1,alg.getClippingOffsetRow(10));
		assertEquals(-2,alg.getClippingOffsetRow(11));
	}

	@Test
	public void getClippingOffsetCol() {
		ClippedPadding2D_F64 alg = createAlg();

		Tensor_F64 tensor = TensorFactory_F64.randomMM(rand,false,-1,1,    4,3,6,2);
		alg.setInput(tensor);

		assertEquals(1,alg.getClippingOffsetCol(0));
		assertEquals(0,alg.getClippingOffsetCol(1));
		assertEquals(0,alg.getClippingOffsetCol(3));
		assertEquals(-1,alg.getClippingOffsetCol(4));
		assertEquals(-2,alg.getClippingOffsetCol(5));
	}

	/**
	 * See if the inner values are returned correctly.  Should be the same as the input just offset
	 */
	@Test
	public void innerValue() {
		ClippedPadding2D_F64 alg = createAlg();

		Tensor_F64 tensor = TensorFactory_F64.randomMM(rand,false,-1,1,    4,3,6,2);

		alg.setInput(tensor);

		// inside the physical tensor
		assertEquals(tensor.get(1,2,1,0),alg.get(1,2,4,1), DeepBoofConstants.TEST_TOL_F64);
	}

	private ClippedPadding2D_F64 createAlg() {
		ConfigPadding config = new ConfigPadding();
		config.x0=1;
		config.x1=2;
		config.y0=3;
		config.y1=4;
		config.type = PaddingType.CLIPPED;

		return new ClippedPadding2D_F64(config);
	}
}