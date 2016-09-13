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

import deepboof.Tensor;
import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialConvolve2D;
import deepboof.forward.SpatialPadding2D;

/**
 * Common class for implementations of {@link SpatialConvolve2D}.  Does not assume how kernels span input channels.
 *
 * @author Peter Abeles
 */
public abstract class BaseSpatialWindow
		<T extends Tensor<T>, P extends SpatialPadding2D<T>>
		extends BaseFunction<T> {

	/**
	 * Configuration for convolution
	 */
	protected ConfigSpatial config;

	// see variable definitions in SpacialTensor2D javadoc
	protected int N,C,H,W; // mini-batch size, input channels, input height, input width
	protected int HH,WW; // kernel height, kernel width

	protected int Ho,Wo;   // output. height and width
	protected int Hp,Wp;   // input + padding, height and width.

	// applies padding to input tensor
	protected P padding;

	public BaseSpatialWindow(ConfigSpatial config, P padding) {
		this.config = config;
		this.padding = padding;
	}

	@Override
	public void _initialize() {
		if( shapeInput.length != 3 )
			throw new IllegalArgumentException("Expected 3D spatial tensor");

		C = shapeInput[0];
		H = shapeInput[1];
		W = shapeInput[2];

		WW = config.WW;
		HH = config.HH;

		int[] paddedShape = padding.shapeGivenInput(shapeInput);

		Hp = paddedShape[1];
		Wp = paddedShape[2];

		if( WW > Wp )
			throw new IllegalArgumentException("Window size is bigger then padded tensor's width");
		if( HH > Hp )
			throw new IllegalArgumentException("Window size is bigger then padded tensor's height");

		Ho = 1 + (Hp - HH) / config.periodY;
		Wo = 1 + (Wp - WW) / config.periodX;

		if( Ho <= 0 )
			throw new IllegalArgumentException("As configured output height is <= 0");
		if( Wo <= 0 )
			throw new IllegalArgumentException("As configured output width is <= 0");

		shapeOutput = new int[]{C,Ho,Wo};
	}

	/**
	 * Do all regions interact with the image border?  The provided point is the
	 * lower-extent of the output pixel which doesn't interact with input image border.
	 *
	 * @param outR0 Lower-extent row in output coordinates.
	 * @param outC0 Lower-extent column in output coordinates.
	 */
	protected boolean isEntirelyBorder(int outR0, int outC0 ) {
		if( outC0 >= Wo || outR0 >= Ho )
			return true;
		else {
			int padR0 = padding.getPaddingRow0();
			int padC0 = padding.getPaddingCol0();

			// see if the first region which could be inside actually goes outside
			if (outC0 * config.periodX + WW > W + padC0 || outR0 * config.periodY + HH > H + padR0)
				return true;
		}
		return false;
	}

	/**
	 * The lower extent in output coordinates for regions that are contained entirely inside the
	 * original image.
	 *s
	 * @param period Sampling period in input tensor pixels
	 * @param padding Padding added along lower extent to input tensor
	 * @return Lower extent in output tensor coordinates, inclusive
	 */
	public static int innerLowerExtent(int period, int padding) {
		return (int)Math.ceil(padding/(double)period);
	}

	/**
	 * The upper extent int output coordinates for regions that are contained entirely inside the
	 * original image.
	 *
	 * @param windowLength Length of sampling window in input tensor pixel
	 * @param period Sampling period in input tensor pixels
	 * @param padding Padding added along lower extent to input tensor
	 * @param inputLength Length of the input tensor
	 * @return Upper extent in output tensor coordinates, exclusive
	 */
	public static int innerUpperExtent( int windowLength , int period , int padding , int inputLength ) {
		return (inputLength+padding-windowLength)/period;
	}

	public P getPadding() {
		return padding;
	}

	//	public ConfigSpatial getConfiguration() {
//		return config;
//	}
}
