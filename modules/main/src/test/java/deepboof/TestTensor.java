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

package deepboof;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Peter Abeles
 */
public class TestTensor {
	@Test
	public void reshape_N() {
		Dummy tensor = new Dummy();

		assertFalse(tensor.subtensor);

		tensor.reshape(new int[]{10});
		assertEquals(1,tensor.getDimension());
		assertEquals(10,tensor.length());
		assertTrue(10 <= tensor.data.length);

		tensor.reshape(new int[]{10,5});
		assertEquals(2,tensor.getDimension());
		assertEquals(50,tensor.length());
		assertTrue(50 <= tensor.data.length);

		tensor.reshape(new int[]{1,2,3});
		assertEquals(3,tensor.getDimension());
		assertEquals(6,tensor.length());
		assertTrue(6 <= tensor.data.length);
		assertEquals(1,tensor.getShape()[0]);
		assertEquals(2,tensor.getShape()[1]);
		assertEquals(3,tensor.getShape()[2]);

		tensor.reshape(new int[]{5});
		assertEquals(1,tensor.getDimension());
		assertEquals(5,tensor.length());
		assertTrue(5 <= tensor.data.length);
		assertEquals(5,tensor.getShape()[0]);
		assertEquals(1,tensor.getShape().length);
	}

	@Test
	public void reshape_1() {
		Dummy tensor = new Dummy();

		tensor.reshape(10);
		assertEquals(1,tensor.getDimension());
		assertEquals(10,tensor.length());
		assertTrue(10 <= tensor.data.length);
		assertEquals(10,tensor.getShape()[0]);
	}

	@Test
	public void reshape_2() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,2);
		assertEquals(2,tensor.getDimension());
		assertEquals(20,tensor.length());
		assertTrue(20 <= tensor.data.length);
		assertEquals(10,tensor.getShape()[0]);
		assertEquals(2,tensor.getShape()[1]);
	}

	@Test
	public void reshape_3() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,2,3);
		assertEquals(3,tensor.getDimension());
		assertEquals(60,tensor.length());
		assertTrue(60 <= tensor.data.length);
		assertEquals(10,tensor.getShape()[0]);
		assertEquals(2,tensor.getShape()[1]);
		assertEquals(3,tensor.getShape()[2]);
	}

	@Test
	public void reshape_4() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,2,3,4);
		assertEquals(4,tensor.getDimension());
		assertEquals(60*4,tensor.length());
		assertTrue(60*4 <= tensor.data.length);
		assertEquals(10,tensor.getShape()[0]);
		assertEquals(2,tensor.getShape()[1]);
		assertEquals(3,tensor.getShape()[2]);
		assertEquals(4,tensor.getShape()[3]);
	}

	@Test
	public void reshape_5() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,2,3,4,1);
		assertEquals(5,tensor.getDimension());
		assertEquals(60*4,tensor.length());
		assertTrue(60*4 <= tensor.data.length);
		assertEquals(10,tensor.getShape()[0]);
		assertEquals(2,tensor.getShape()[1]);
		assertEquals(3,tensor.getShape()[2]);
		assertEquals(4,tensor.getShape()[3]);
		assertEquals(1,tensor.getShape()[4]);
	}

	@Test
	public void idx_N() {
		Dummy tensor = new Dummy();

		tensor.reshape(10);
		assertEquals(2,tensor.idx(new int[]{2}));


		tensor.reshape(10,2,6);
		assertEquals(2*2*6 + 1*6 + 4,tensor.idx(new int[]{2,1,4}));

		tensor.startIndex = 5;
		assertEquals(5+2*2*6 + 1*6 + 4,tensor.idx(new int[]{2,1,4}));
	}

	@Test
	public void idx_1() {
		Dummy tensor = new Dummy();

		tensor.reshape(10);
		assertEquals(2,tensor.idx(2));

		tensor.startIndex = 5;
		assertEquals(5+2,tensor.idx(2));
	}

	@Test
	public void idx_2() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5);
		assertEquals(2*5+3,tensor.idx(2,3));

		tensor.startIndex = 5;
		assertEquals(5+2*5+3,tensor.idx(2,3));
	}

	@Test
	public void idx_3() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6);
		assertEquals((2*5+3)*6+2,tensor.idx(2,3,2));

		tensor.startIndex = 5;
		assertEquals(5+(2*5+3)*6+2,tensor.idx(2,3,2));
	}

	@Test
	public void idx_4() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7);
		assertEquals(((2*5+3)*6+2)*7+5,tensor.idx(2,3,2,5));

		tensor.startIndex = 5;
		assertEquals(5+((2*5+3)*6+2)*7+5,tensor.idx(2,3,2,5));
	}

	@Test
	public void idx_5() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7,3);
		assertEquals((((2*5+3)*6+2)*7+5)*3+2,tensor.idx(2,3,2,5,2));

		tensor.startIndex = 5;
		assertEquals(5+(((2*5+3)*6+2)*7+5)*3+2,tensor.idx(2,3,2,5,2));
	}

	@Test
	public void computeStrides() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7);
		tensor.computeStrides();
		assertEquals(5*6*7,tensor.strides[0]);
		assertEquals(6*7,tensor.strides[1]);
		assertEquals(7,tensor.strides[2]);
		assertEquals(1,tensor.strides[3]);

	}

	@Test
	public void stride() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7);
		tensor.computeStrides();

		for (int i = 0; i < 4; i++) {
			assertEquals(tensor.strides[i],tensor.stride(i));
		}

		for (int i = 0; i < 4; i++) {
			assertEquals(tensor.strides[4-i-1],tensor.stride(-i-1));
		}
	}

	@Test
	public void isShape() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7);

		int[] shape = new int[]{10,5,6,7};

		assertTrue(tensor.isShape(10,5,6,7));

		for (int i = 0; i < shape.length; i++) {
			int before = shape[i];
			shape[i] = before+1;
			assertFalse(tensor.isShape(shape));
			shape[i] = before;
		}

		Dummy foo = new Dummy();
		foo.reshape(10,5,6,7);

		assertTrue(tensor.isShape(foo.getShape()));
	}

	@Test
	public void getLength() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6,7);
		tensor.reshape(3,1,4);

		assertEquals(12,tensor.length());

	}

	@Test
	public void indexToCoor() {
		Dummy tensor = new Dummy();

		tensor.reshape(10,5,6);

		int index = tensor.idx(3,4,1);
		int[] found = tensor.indexToCoor(index,null);

		assertEquals(3,found.length);
		assertEquals(3,found[0]);
		assertEquals(4,found[1]);
		assertEquals(1,found[2]);
	}


	@Test
	public void setTo() {
		Dummy tensor = new Dummy();
		tensor.reshape(3,2,4);
		tensor.data[5] = 7;

		Dummy found = new Dummy();
		found.reshape(3,2,4);
		found.setTo(tensor);

		assertTrue(tensor.isShape(found.getShape()));
		assertTrue(7 == found.data[5]);
	}

	@Test
	public void setTo_sub() {
		// create a sub tensor by adding an offset
		Dummy tensor = new Dummy();
		tensor.reshape(3,2,4);
		tensor.data[5] = 7;
		tensor.reshape(3,2);
		tensor.startIndex = 2;

		// set the dummy to the tensor, this should shift the 7 because found doesn't have
		// an offset
		Dummy found = new Dummy();
		found.setTo(tensor);

		assertTrue(tensor.isShape(found.getShape()));
		assertTrue(7 == found.data[3]);
	}

	@Test
	public void subtensor() {
		Dummy tensor = new Dummy();
		tensor.reshape(3,6,8);

		Dummy sub = (Dummy)tensor.subtensor(5,new int[]{2,5});

		assertEquals(true,sub.subtensor);
		assertEquals(5,sub.startIndex);
		assertEquals(2,sub.shape.length);
		assertEquals(2,sub.shape[0]);
		assertEquals(5,sub.shape[1]);

	}

	static class Dummy extends Tensor {

		double data[] = new double[0];

		@Override
		public double getDouble(int... coordinate) {
			return 0;
		}

		@Override
		public Object getData() {
			return data;
		}

		@Override
		public void setData(Object data) {
			this.data = (double[])data;
		}

		@Override
		protected void innerArrayGrow(int N) {
			data = new double[N];
		}

		@Override
		protected int innerArrayLength() {
			return data.length;
		}

		@Override
		public Tensor create(int[] shape) {
			return new Dummy();
		}

		@Override
		public void zero() {}

		@Override
		public Class getDataType() {
			return double.class;
		}
	}
}