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


import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;

/**
 * @author Peter Abeles
 */
public class DeepUnitTest {
	public static void assertEquals(int[] expected, int[] found) {
		org.junit.Assert.assertEquals(found.length,expected.length);
		for (int i = 0; i < expected.length; i++) {
			org.junit.Assert.assertEquals(expected[i],found[i]);
		}
	}

	public static<T extends Tensor>
	void assertNotEquals(T expected , T found , Accuracy tol ) {
		if( expected instanceof Tensor_F64 ) {
			assertNotEquals((Tensor_F64)expected,(Tensor_F64)found,tol.value(double.class));
		} else {
			assertNotEquals((Tensor_F32)expected,(Tensor_F32)found,(float)tol.value(float.class));
		}
	}

	public static<T extends Tensor>
	void assertEquals(T expected , T found , Accuracy tol ) {
		if( expected instanceof Tensor_F64 ) {
			assertEquals((Tensor_F64)expected,(Tensor_F64)found,tol.value(double.class));
		} else {
			assertEquals((Tensor_F32)expected,(Tensor_F32)found,(float)tol.value(float.class));
		}
	}

	public static void assertEquals(Tensor_F64 expected , Tensor_F64 found , double tol ) {
		TensorOps.checkShape("foo",-1,expected.getShape(),found.getShape(),false);

		int indexE = expected.startIndex;
		int indexF = found.startIndex;

		int N = expected.length();

		for (int i = 0; i < N; i++) {
			org.junit.Assert.assertEquals("i = "+i+" indexes "+indexE+" "+indexF,expected.d[indexE++], found.d[indexF++], tol);
		}
	}

	public static void assertEquals(Tensor_F32 expected , Tensor_F32 found , float tol ) {
		TensorOps.checkShape("foo",-1,expected.getShape(),found.getShape(),false);

		int indexE = expected.startIndex;
		int indexF = found.startIndex;

		int N = expected.length();

		for (int i = 0; i < N; i++) {
			org.junit.Assert.assertEquals(indexE+" "+indexF,expected.d[indexE++], found.d[indexF++], tol);
		}
	}

	public static void assertNotEquals(Tensor_F64 expected , Tensor_F64 found , double tol ) {
		TensorOps.checkShape("foo",-1,expected.getShape(),found.getShape(),false);

		int indexE = expected.startIndex;
		int indexF = found.startIndex;

		int N = expected.length();

		for (int i = 0; i < N; i++) {
			org.junit.Assert.assertNotEquals("i = "+i+" indexes "+indexE+" "+indexF,expected.d[indexE++], found.d[indexF++], tol);
		}
	}

	public static void assertNotEquals(Tensor_F32 expected , Tensor_F32 found , float tol ) {
		TensorOps.checkShape("foo",-1,expected.getShape(),found.getShape(),false);

		int indexE = expected.startIndex;
		int indexF = found.startIndex;

		int N = expected.length();

		for (int i = 0; i < N; i++) {
			org.junit.Assert.assertNotEquals(indexE+" "+indexF,expected.d[indexE++], found.d[indexF++], tol);
		}
	}
}
