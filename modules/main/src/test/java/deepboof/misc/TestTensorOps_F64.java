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
import deepboof.DeepUnitTest;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * @author Peter Abeles
 */
public class TestTensorOps_F64 {

	Random rand = new Random(234);

	@Test
	public void elementMult_scalar() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 T = TensorFactory_F64.random(rand,sub, 5,3,1);

			Tensor_F64 original = T.copy();

			double s = 2.1;
			TensorOps_F64.elementMult(T,s);

			int N = original.length();
			for (int i = 0; i < N; i++) {
				assertEquals( original.getAtIndex(i)*s , T.getAtIndex(i) , DeepBoofConstants.TEST_TOL_F64 );
			}
		}
	}

	@Test
	public void elementMult_scalar_tensor() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 T = TensorFactory_F64.random(rand,sub, 5,3,1);
			Tensor_F64 O = TensorFactory_F64.random(rand,sub, 5,3,1);

			Tensor_F64 original = T.copy();

			double s = 2.1;
			TensorOps_F64.elementMult(T,s,O);

			int N = original.length();
			for (int i = 0; i < N; i++) {
				assertEquals( original.getAtIndex(i) , T.getAtIndex(i) , DeepBoofConstants.TEST_TOL_F64 );
				assertEquals( original.getAtIndex(i)*s , O.getAtIndex(i) , DeepBoofConstants.TEST_TOL_F64 );
			}
		}
	}

	@Test
	public void elementMult_tensor() {
		List<int[]> shapes = new ArrayList<>();
		shapes.add(new int[]{5});
		shapes.add(new int[]{2,3,4});

		for( boolean subtensor : new boolean[]{false,true}) {
			for( int[] shape : shapes ) {
				Tensor_F64 A = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 B = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 found = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 expected = new Tensor_F64(shape);

				for (int i = 0; i < A.length(); i++) {
					expected.d[expected.startIndex+i] = A.d[A.startIndex+i]*B.d[B.startIndex+i];
				}

				TensorOps_F64.elementMult(A,B,found);

				DeepUnitTest.assertEquals(expected,found,DeepBoofConstants.TEST_TOL_F64);
			}
		}
	}

	@Test
	public void elementAdd_tensor() {
		List<int[]> shapes = new ArrayList<>();
		shapes.add(new int[]{5});
		shapes.add(new int[]{2,3,4});

		for( boolean subtensor : new boolean[]{false,true}) {
			for( int[] shape : shapes ) {
				Tensor_F64 A = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 B = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 found = TensorFactory_F64.random(rand, subtensor, shape);
				Tensor_F64 expected = new Tensor_F64(shape);

				for (int i = 0; i < A.length(); i++) {
					expected.d[expected.startIndex+i] = A.d[A.startIndex+i] + B.d[B.startIndex+i];
				}

				TensorOps_F64.elementAdd(A,B,found);

				DeepUnitTest.assertEquals(expected,found,DeepBoofConstants.TEST_TOL_F64);
			}
		}
	}

	@Test
	public void elementSum() {
		List<int[]> shapes = new ArrayList<>();
		shapes.add(new int[]{5});
		shapes.add(new int[]{2,3,4});

		for( boolean subtensor : new boolean[]{false,true}) {
			for( int[] shape : shapes ) {
				Tensor_F64 tensor = TensorFactory_F64.random(rand, subtensor, shape);

				double expected = 0;
				for (int i = 0; i < tensor.length(); i++) {
					expected += tensor.getAtIndex(i);
				}

				double found = TensorOps_F64.elementSum(tensor);

				assertEquals(expected,found,DeepBoofConstants.TEST_TOL_F64);
			}
		}
	}

	@Test
	public void insertSubChannel() {
		for( boolean subtensor : new boolean[]{false,true}) {

			Tensor_F64 src = TensorFactory_F64.random(rand, subtensor, 4, 10, 8);
			Tensor_F64 dst = TensorFactory_F64.random(rand, subtensor, 5, 11, 12);

			TensorOps_F64.insertSubChannel(src,15,7,dst,19,8,3,4);

			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 4; col++) {
					assertEquals( src.d[15+row*7+col],dst.d[19+row*8+col], DeepBoofConstants.TEST_TOL_F64);
				}
			}
		}
	}

	@Test
	public void insertSpatial() {
		for( boolean subtensor : new boolean[]{false,true}) {
			int C = 3;
			int H = 10;
			int W = 15;

			Tensor_F64 src = TensorFactory_F64.random(rand, subtensor, C, H, W);
			Tensor_F64 dst = TensorFactory_F64.random(rand, subtensor, C, H, W);

			int[] coorSrc = new int[3];
			int[] coorDst = new int[3];

			TensorOps_F64.insertSpatial(src, coorSrc, dst, coorDst);
			checkSpatial(src, coorSrc, dst, coorDst);

			// increase the dimensions in dst
			dst = TensorFactory_F64.random(rand, subtensor, 10, C, H, W);
			coorDst = new int[]{5, 0, 0, 0};

			TensorOps_F64.insertSpatial(src, coorSrc, dst, coorDst);
			checkSpatial(src, coorSrc, dst, coorDst);

			// increase the dimension in src
			src = TensorFactory_F64.random(rand, subtensor, 4, C, H, W);
			coorSrc = new int[]{1, 0, 0, 0};

			TensorOps_F64.insertSpatial(src, coorSrc, dst, coorDst);
			checkSpatial(src, coorSrc, dst, coorDst);

			// paste it inside dst, which is now larger than src
			dst = TensorFactory_F64.random(rand, subtensor, 10, C, H+4, W+6);
			coorDst = new int[]{5, 0, 1, 2};

			TensorOps_F64.insertSpatial(src, coorSrc, dst, coorDst);
			checkSpatial(src, coorSrc, dst, coorDst);

			// Swap width and height
			src = TensorFactory_F64.random(rand, subtensor, C, W, H);
			dst = TensorFactory_F64.random(rand, subtensor, 10, C, W+6, H+4);
			coorSrc = new int[3];
			coorDst = new int[]{5, 0, 1, 2};

			TensorOps_F64.insertSpatial(src, coorSrc, dst, coorDst);
			checkSpatial(src, coorSrc, dst, coorDst);
		}
	}

	private void checkSpatial( Tensor_F64 src , int[]coorSrc , Tensor_F64 dst , int[]coorDst ) {
		int C = src.length(-3);
		int H = src.length(-2);
		int W = src.length(-1);

		int a = coorSrc.length-3;
		int b = coorDst.length-3;

		int y0 = coorDst[b+1];
		int x0 = coorDst[b+2];


		for (int c = 0; c < C; c++) {
			coorSrc[a] = c;
			coorDst[b] = c;

			for (int y = 0; y < H; y++) {
				coorSrc[a+1] = y;
				coorDst[b+1] = y0 + y;

				for (int x = 0; x < W; x++) {
					coorSrc[a+2] = x;
					coorDst[b+2] = x0 + x;

					int indexSrc = src.idx(coorSrc);
					int indexDst = dst.idx(coorDst);

					assertEquals(src.d[indexSrc],dst.d[indexDst],DeepBoofConstants.TEST_TOL_F64);
				}
			}
		}
	}

	@Test
	public void fillSpatialBorder() {
		for( boolean subtensor : new boolean[]{false,true}) {
			int C = 3;
			int H = 13;
			int W = 18;
			double value = 2.3;
			int borderX0 = 2, borderY0 = 1;
			int borderX1 = 3, borderY1 = 4;

			int[] coor = new int[]{2,0,0,0};

			Tensor_F64 T = TensorFactory_F64.random(rand, subtensor,5, C, H, W);
			TensorOps_F64.fillSpatialBorder(T,coor,borderY0,borderX0,borderY1,borderX1,value);
			checkBorder(T,2,borderY0,borderX0,borderY1,borderX1, value);

			// see if it ignores the last parts of the coordinate
			coor = new int[]{2,10,2,12};

			T = TensorFactory_F64.random(rand, subtensor,5, C, H, W);
			TensorOps_F64.fillSpatialBorder(T,coor,borderY0,borderX0,borderY1,borderX1,value);
			checkBorder(T,2,borderY0,borderX0,borderY1,borderX1, value);
		}
	}

	private void checkBorder( Tensor_F64 T , int batch ,
							  int borderY0, int borderX0, int borderY1, int borderX1,
							  double value )
	{
		int C = T.length(1);
		int H = T.length(2);
		int W = T.length(3);

		int[] coor = new int[]{batch,0,0,0};
		for (int c = 0; c < C; c++) {
			coor[1] = c;
			for (int y = 0; y < H; y++) {
				coor[2] = y;
				for (int x = 0; x < W; x++) {
					coor[3] = x;
					if( y < borderY0 || y >= H-borderY1|| x < borderX0 || x >= W-borderX1 )
						assertEquals(x+" "+y,value,T.get(coor),DeepBoofConstants.TEST_TOL_F64);
				}
			}
		}
	}

	@Test
	public void fill() {
		Tensor_F64 a = new Tensor_F64(1,2,3,4);

		TensorOps_F64.fill(a,2.0);

		for (int i = 0; i < a.d.length; i++) {
			assertEquals(2.0,a.d[i], DeepBoofConstants.TEST_TOL_F64);
		}

		// try it with a sub-matrix now
		a = a.subtensor(5,new int[]{4});
		TensorOps_F64.fill(a,3.0);

		for (int i = 0; i < 5; i++) {
			assertEquals(2.0,a.d[i], DeepBoofConstants.TEST_TOL_F64);
		}
		for (int i = 0; i < 4; i++) {
			assertEquals(3.0,a.d[a.startIndex+i], DeepBoofConstants.TEST_TOL_F64);
		}
	}

}
