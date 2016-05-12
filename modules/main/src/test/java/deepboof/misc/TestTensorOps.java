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

import deepboof.Tensor;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * @author Peter Abeles
 */
public class TestTensorOps {
	@Test
	public void sumTensorLength() {
		List<int[]> shapes = new ArrayList<int[]>();

		assertEquals(0, TensorOps.sumTensorLength(shapes));
		shapes.add(new int[]{3});
		assertEquals(3, TensorOps.sumTensorLength(shapes));
		shapes.add(new int[]{2,5,1});
		assertEquals(13, TensorOps.sumTensorLength(shapes));
	}

	@Test
	public void tensorLength() {
		assertEquals(0, TensorOps.tensorLength());
		assertEquals(0, TensorOps.tensorLength(5,0));
		assertEquals(5, TensorOps.tensorLength(5,1));
		assertEquals(5, TensorOps.tensorLength(5));
		assertEquals(12, TensorOps.tensorLength(2,3,2));
	}

	@Test
	public void checkShape_list() {
		List<int[]> expected = new ArrayList<int[]>();
		List<Tensor<?>> actual = new ArrayList();

		expected.add(new int[]{2,1,3});
		actual.add(new Tensor_F64(2,1,3));

		TensorOps.checkShape("",expected,actual,false);

		List<Tensor<?>> actual2 = new ArrayList();
		actual2.add(new Tensor_F64(2,2,1,3));
		TensorOps.checkShape("",expected,actual2,true);

		expected.add(new int[]{5,1,4});
		actual.add(new Tensor_F64(2,1,4));

		try {
			TensorOps.checkShape("",expected,actual,false);
			fail("Should have thrown an exception");
		} catch( IllegalArgumentException ignore){}

		expected.add(new int[]{5,1,4});
		actual2.add(new Tensor_F64(2,2,1,4));

		try {
			TensorOps.checkShape("",expected,actual,true);
			fail("Should have thrown an exception");
		} catch( IllegalArgumentException ignore){}
	}

	@Test
	public void checkShape_individual() {
		TensorOps.checkShape("",2,new int[]{2,1,3},new int[]{2,1,3},false);
		TensorOps.checkShape("",2,new int[]{2,1,3},new int[]{10,2,1,3},true);

		try {
			TensorOps.checkShape("",2,new int[]{2,2,3},new int[]{2,1,3},false);
			fail("Should have thrown an exception");
		} catch( IllegalArgumentException ignore){}

		try {
			TensorOps.checkShape("",2,new int[]{2,2,3},new int[]{10,2,1,3},false);
			fail("Should have thrown an exception");
		} catch( IllegalArgumentException ignore){}
	}

	@Test
	public void outerLength() {
		int shape[] = new int[]{5,2,3,2};

		assertEquals(2, TensorOps.outerLength(shape,3));
		assertEquals(6, TensorOps.outerLength(shape,2));
		assertEquals(12, TensorOps.outerLength(shape,1));
		assertEquals(60, TensorOps.outerLength(shape,0));
	}
}
