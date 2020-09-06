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

package deepboof.tensors;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Peter Abeles
 */
public class TestTensor_U8 {
	@Test
	public void reshapeInnerArray() {
		Tensor_U8 T = new Tensor_U8(2,6);

		assertEquals(12,T.d.length);
		T.reshape(2,1,2);
		assertEquals(12,T.d.length);
		T.reshape(5,6,2);
		assertEquals(5*6*2,T.d.length);
	}

	@Test
	public void create() {
		Tensor_U8 T = new Tensor_U8();

		Tensor_U8 F = T.create(2,5);

		assertFalse(F.subtensor);
		assertTrue(F.isShape(2,5));
	}

	@Test
	public void getDataType() {
		Tensor_U8 T = new Tensor_U8();

		assertTrue(byte.class == T.getDataType());
	}

	@Test
	public void getInnerLength() {
		Tensor_U8 T = new Tensor_U8(3,4,1);
		T.reshape(2,2);

		assertEquals(12,T.innerArrayLength());
	}
}