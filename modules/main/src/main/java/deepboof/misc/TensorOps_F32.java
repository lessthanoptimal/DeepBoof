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

import deepboof.tensors.Tensor_F32;

import java.util.Arrays;

/**
 * @author Peter Abeles
 */
public class TensorOps_F32 {

	/**
	 * Performs an element-wise scalar multiplication on the tensor
	 * @param tensor Tensor which is multiplied
	 * @param value value of the multiplication
	 */
	public static void elementMult(Tensor_F32 tensor , float value  ) {
		int index = tensor.startIndex;
		int end = index + tensor.length();
		while( index < end ) {

			tensor.d[index++] *= value;
		}
	}

	/**
	 * Performs an element-wise scalar multiplication
	 *
	 * @param input Tensor which is multiplied (not modified)
	 * @param value value of the multiplication (modified)
	 * @param output Tensor where the results are stored
	 */
	public static void elementMult(Tensor_F32 input , float value , Tensor_F32 output ) {
		TensorOps.checkShape(input,output);

		int indexIn = input.startIndex;
		int indexOut = output.startIndex;
		int end = indexIn + input.length();
		for( ; indexIn < end; indexIn++ , indexOut++ ) {
			output.d[indexOut] = input.d[indexIn]*value;
		}
	}

	/**
	 * <p>Performs element-wise multiplication between the two tensors and stores results in output.  All tensors
	 * must have the same shape.  </p>
	 *
	 * {@code output[i] = A[i]*B[i]}
	 *
	 * @param A Input tensor.  Can be the same as output.
	 * @param B Input tensor.  Can be the same as output.
	 * @param output Output tensor.
	 */
	public static void elementMult(Tensor_F32 A , Tensor_F32 B , Tensor_F32 output ) {
		int indexA = A.startIndex;
		int endA = indexA + A.length();
		int indexB = B.startIndex;

		if( A != output && B != output ) {

			int indexOut = output.startIndex;

			while (indexA < endA) {
				output.d[indexOut++] = A.d[indexA++] * B.d[indexB++];
			}
		} else if( B == output ) {
			while (indexA < endA) {
				B.d[indexB++] *= A.d[indexA++];
			}
		} else if( A == output ) {
			while (indexA < endA) {
				A.d[indexA++] *= B.d[indexB++];
			}
		}
	}

	/**
	 * <p>Performs element-wise addition between the two tensors and stores results in output.  All tensors
	 * must have the same shape.  </p>
	 *
	 * {@code output[i] = A[i] + B[i]}
	 *
	 * @param A Input tensor.  Can be the same as output.
	 * @param B Input tensor.  Can be the same as output.
	 * @param output Output tensor.
	 */
	public static void elementAdd(Tensor_F32 A , Tensor_F32 B , Tensor_F32 output ) {
		int indexA = A.startIndex;
		int endA = indexA + A.length();
		int indexB = B.startIndex;

		if( A != output && B != output ) {

			int indexOut = output.startIndex;

			while (indexA < endA) {
				output.d[indexOut++] = A.d[indexA++] + B.d[indexB++];
			}
		} else if( B == output ) {
			while (indexA < endA) {
				B.d[indexB++] += A.d[indexA++];
			}
		} else {
			while (indexA < endA) {
				A.d[indexA++] += B.d[indexB++];
			}
		}
	}

	/**
	 * Computes the sum of all the elements in the tensor
	 * @param tensor Tensor
	 */
	public static float elementSum( Tensor_F32 tensor ) {
		int index = tensor.startIndex;
		int end = index + tensor.length();

		float sum = 0;
		while( index < end ) {
			sum += tensor.d[index++];
		}
		return sum;
	}

	/**
	 * Used to copy a sub-image between two image tensors.
	 *
	 * @param src Source tensor
	 * @param srcStartIndex Start index in input tensor.
	 * @param srcStride Row-stride for input tensor
	 * @param dst Destination tensor
	 * @param dstStartIndex Start index in destination tensor.
	 * @param dstStride Row-stride for destination tensor
	 * @param rows Number of rows to be copied
	 * @param columns Number of columns to be copied
	 */
	public static void insertSubChannel( Tensor_F32 src , int srcStartIndex , int srcStride ,
										 Tensor_F32 dst , int dstStartIndex , int dstStride ,
										 int rows , int columns )
	{
		int indexSrc = srcStartIndex;
		int indexDst = dstStartIndex;;

		for (int i = 0; i < rows; i++) {
			System.arraycopy(src.d,indexSrc,dst.d,indexDst,columns);

			indexSrc += srcStride;
			indexDst += dstStride;
		}
	}

	/**
	 * Inserts the spatial region of one tensor into another.  Both tensors are assumed
	 * to follow the following pattern for their shape.  (..., C, H, W).  C is for
	 * the number of channels, H is for the image's height, and W, is for the image's width.
	 *
	 * @param src Source tensor.  Entire image is copied into dst.
	 * @param srcCoor Coordinate of spatial region.  ( ..., 0, 0, 0) modified.
	 * @param dst Destination tensor.  The source image can be smaller than the destination, but not larger.
	 * @param dstCoor Coordinate of spatial region.  ( ..., 0, y, x) modified.
	 */
	public static void insertSpatial(Tensor_F32 src , int []srcCoor,
									 Tensor_F32 dst , int []dstCoor )
	{
		if( srcCoor.length < 3) throw new IllegalArgumentException("dimensions must be >= 3 for src");
		if( dstCoor.length < 3) throw new IllegalArgumentException("dimensions must be >= 3 for dst");
		if( srcCoor.length != src.getDimension() )
			throw new IllegalArgumentException("Coordinate length doesn't match tensor dimension for src");
		if( dstCoor.length != dst.getDimension() )
			throw new IllegalArgumentException("Coordinate length doesn't match tensor dimension for dst");

		// axis of spatial region channel
		int srcAxis = srcCoor.length-3;
		int dstAxis = dstCoor.length-3;

		// enforce constraint on coordinates
		for (int i = 0; i < 3; i++) {
			srcCoor[srcAxis+i] = 0;
		}
		dstCoor[dstAxis] = 0;

		// Shape of spatial region
		int numChannels = src.length(-3);
		int height = src.length(-2);
		int width = src.length(-1);

		int heightDst = dst.length(-2);
		int widthDst = dst.length(-1);

		// sanity checks
		if( numChannels != dst.length(dstAxis)) {
			throw new IllegalArgumentException("Number of channels do not match in src and dst");
		}

		if( height > heightDst ) {
			throw new IllegalArgumentException("src height is larger than dst");
		}
		if( width > widthDst ) {
			throw new IllegalArgumentException("src width is larger than dst");
		}

		// get ready for copy
		int pixelSrc = src.idx(srcCoor);
		int pixelDst = dst.idx(dstCoor);

		if( width == widthDst && height == heightDst )
			System.arraycopy(src.d,pixelSrc,dst.d,pixelDst,numChannels*width*height);
		else {
			int channelSrc = pixelSrc;
			int channelDst = pixelDst;

			for (int channel = 0; channel < numChannels; channel++) {
				pixelSrc = channelSrc;
				pixelDst = channelDst;

				for (int row = 0; row < height; row++) {
					System.arraycopy(src.d,pixelSrc,dst.d,pixelDst,width);
					pixelSrc += width;
					pixelDst += widthDst;
				}
				channelSrc += width*height;
				channelDst += widthDst*heightDst;
			}
		}

	}

	/**
	 * Fills the border with the specified value.  The tensor is assumed to have the following
	 * shape ( ... , C , H, W),  C is for he number of channels, H is for the image's height, and W, is for
	 * the image's width.
	 * 
	 * @param tensor Tensor with a spatial region at the end. Modified.
	 * @param coor Coordinate of the targeted spatial region inside the tensor.  axis values for C,H,W are ignored. Modified.
	 * @param borderY0 Lower extent's border length along Y axis
	 * @param borderX0 Lower extent's border length along X axis
	 * @param borderY1 Upper extent's border length along Y axis
	 * @param borderX1 Upper extent's border length along X axis
	 * @param value Value that is to be inserted
	 */
	public static void fillSpatialBorder(Tensor_F32 tensor , int[] coor ,
										 int borderY0 , int borderX0 , int borderY1 , int borderX1 ,
										 float value ) {

		// axis of spatial region channel
		int channelAxis = coor.length-3;

		// By zeroing the spatial portion it will return the index of the spatial region's start
		for (int i = channelAxis; i < coor.length; i++) {
			coor[i] = 0;
		}
		
		int numChannels = tensor.length(channelAxis);
		int height = tensor.length(channelAxis+1);
		int width  = tensor.length(channelAxis+2);

		if( borderY0+borderY1 > height ) {
			throw new IllegalArgumentException("Y border is larger than image height");
		}
		if( borderX0+borderX1 > width ) {
			throw new IllegalArgumentException("X border is larger than image width");
		}

		// run through each channel and fill the borders
		for (int channel = 0; channel < numChannels; channel++) {
			coor[channelAxis]   = channel;
			coor[channelAxis+1] = 0;
			coor[channelAxis+2] = 0;

			// fill the top and bottom
			int indexTop = tensor.idx(coor);
			Arrays.fill(tensor.d,indexTop,indexTop+borderY0*width,value);

			coor[channelAxis+1] = height-borderY1;
			int indexBottom = tensor.idx(coor);
			Arrays.fill(tensor.d,indexBottom,indexBottom+borderY1*width,value);

			for (int y = borderY0; y < height - borderY1; y++) {
				coor[channelAxis+1] = y;
				int left  = tensor.idx(coor);
				int right = left + width-borderX1;

				for (int i = 0; i < borderX0; i++) {
					tensor.d[left+i] = value;
				}
				for (int i = 0; i < borderX1; i++) {
					tensor.d[right+i] = value;
				}
			}
		}
	}

	/**
	 * Prints a single batch and channel in a spatial tensor
	 * @param tensor The tensor
	 * @param batch Batch number
	 * @param channel channel
	 */
	public static void printSpatial( Tensor_F32 tensor , int batch , int channel ) {
		int rows = tensor.length(2);
		int cols = tensor.length(3);

		System.out.println(tensor.getClass().getSimpleName()+" batch "+batch+"  channel "+channel );
		System.out.println("     rows "+rows+" columns "+cols);
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				System.out.printf("%10.3fe ",tensor.get(batch,channel,row,col));
			}
			System.out.println();
		}
	}

	/**
	 * Fills the tensor with the specified value
	 * @param tensor The tensor
	 * @param value fill value
	 */
	public static void fill( Tensor_F32 tensor , float value ) {
		Arrays.fill(tensor.d,tensor.startIndex,tensor.startIndex+tensor.length(),value);
	}
}
