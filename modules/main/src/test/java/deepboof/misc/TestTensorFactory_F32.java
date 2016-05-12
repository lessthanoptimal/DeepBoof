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

package deepboof.misc;

import deepboof.DeepBoofConstants;
import deepboof.tensors.Tensor_F32;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * @author Peter Abeles
 */
public class TestTensorFactory_F32 {

	Random rand = new Random(234);

	@Test
	public void zeros() {

		Tensor_F32 tensor = TensorFactory_F32.zeros(null,4,3,2);

		assertEquals(4,tensor.length(0));
		assertEquals(3,tensor.length(1));
		assertEquals(2,tensor.length(2));
		assertFalse(tensor.isSub());

		int N = tensor.length();
		for (int i = 0; i < N; i++) {
			assertEquals(0.0f,tensor.getAtIndex(i), DeepBoofConstants.TEST_TOL_F32);
		}

		tensor = TensorFactory_F32.zeros(rand,4,3,2);

		assertEquals(4,tensor.length(0));
		assertEquals(3,tensor.length(1));
		assertEquals(2,tensor.length(2));
		assertTrue(tensor.isSub());

		for (int i = 0; i < N; i++) {
			assertEquals(0.0f,tensor.getAtIndex(i), DeepBoofConstants.TEST_TOL_F32);
		}
	}

	@Test
	public void randomMM() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F32 tensor = TensorFactory_F32.randomMM(rand, sub, -1,1, 4, 3, 2);

			assertEquals(4, tensor.length(0));
			assertEquals(3, tensor.length(1));
			assertEquals(2, tensor.length(2));
			assertEquals(sub, tensor.isSub());

			int N = tensor.length();
			float sum = 0;
			for (int i = 0; i < N; i++) {
				float v = tensor.getAtIndex(i);
				assertTrue(v != 0);
				sum += v;
			}

			assertTrue(Math.abs(sum) < 4*3*2/2.0f);
		}
	}

}