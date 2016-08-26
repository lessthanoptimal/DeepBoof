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

package deepboof.backward;

import deepboof.DFunction;
import deepboof.DeepBoofConstants;
import deepboof.DeepUnitTest;
import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.impl.backward.standard.DConstantPadding2D_F64;
import deepboof.misc.TensorFactory_F64;
import deepboof.misc.TensorOps;
import deepboof.misc.TensorOps_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertTrue;

/**
 *
 *
 * @author Peter Abeles
 */
public abstract class ChecksBackwards_DSpatialWindow {
	Random rand = new Random(234);

	int pad = 2;

	int N = 3;
	int C = 4;
	ConfigSpatial configSpatial;

	DSpatialPadding2D<Tensor_F64> padding;

	public abstract DFunction<Tensor_F64> create(ConfigSpatial config , DSpatialPadding2D_F64 padding );

	private DSpatialPadding2D_F64 createPadding() {
		ConfigPadding config = new ConfigPadding();
		config.y0 = pad;
		config.x0 = pad;
		config.y1 = pad;
		config.x1 = pad;

		return new DConstantPadding2D_F64(config);
	}

	/**
	 * In this scenario input tensor is small enough and the window large enough that every single region will
	 * touch the image border
	 */
	@Test
	public void isEntirelyBorder() {
		for( boolean sub : new boolean[]{false,true}) {
			pad = 2;
			standardCheck(1,1,5,5,sub);
		}
	}

	/**
	 * There will be a mixture of regions which are entirely inside the original and which touch the border
	 */
	@Test
	public void insideAndOutside() {
		for (boolean sub : new boolean[]{false, true}) {
			// single window no border
			pad = 0;
			standardCheck(3,3,3,3,sub);
			// multiple windows in border and entirely inside
			pad = 2;
			standardCheck(6,7,3,3,sub);
		}
	}

	protected void standardCheck( int inH , int inW , int HH , int WW , boolean sub ) {
		int outH = inH + pad*2 - (HH-1);
		int outW = inW + pad*2 - (WW-1);

		Tensor_F64 original = TensorFactory_F64.random(rand, sub,N,C,inH,inW);

		configSpatial = new ConfigSpatial();
		configSpatial.WW = WW;
		configSpatial.HH = HH;

		// Padding of 'pad' is added to operation
		DFunction<Tensor_F64> helper = create(configSpatial,createPadding());

		helper.initialize(C,inH,inW);

		// the forward pass is expected to be run first before backwards
		Tensor_F64 output = new Tensor_F64(WI(N,C,outH,outW));
		helper.forward(original, output); // ignore the results, those are checked in TestForward_*

		// Now do the backwards pass
		Tensor_F64 gradientInput = new Tensor_F64(WI(N,C,inH,inW));
		Tensor_F64 dout = TensorFactory_F64.random(rand,sub,WI(N,helper.getOutputShape()));

		List<Tensor_F64> gradientParameters = new ArrayList<>();

		helper.backwards(original, dout, gradientInput, gradientParameters);

		compareToExpected(original, dout, gradientInput);
	}

	protected void compareToExpected(Tensor_F64 input , Tensor_F64 dout, Tensor_F64 gradientInput ) {

		DSpatialPadding2D_F64 padding = createPadding();
		padding.setInput(input);

		int periodY = configSpatial.periodY;
		int periodX = configSpatial.periodX;
		int HH = configSpatial.HH;
		int WW = configSpatial.WW;

		int H = input.length(2)+pad*2;
		int W = input.length(3)+pad*2;

		int shapePadded[] = padding.getShape();

		// just the spatial component
		Tensor_F64 gradientPadding = new Tensor_F64(TensorOps.WI(shapePadded[2],shapePadded[3]));
		Tensor_F64 expectedGradientInput = gradientInput.createLike();

		for (int batch = 0; batch < N; batch++) {
			for (int channel = 0; channel < C; channel++) {
				gradientPadding.zero();
				int yout = 0;
				for (int y = 0; y <= H-HH; y += periodY, yout++) {
					int xout = 0;
					for (int x = 0; x <= W - WW; x += periodX, xout++ ) {
						computeGradientPadding(padding,batch,channel, y, x, HH, WW, gradientPadding);
					}
				}
				// sanity check for unit test
				assertTrue(TensorOps_F64.elementSum(gradientPadding) != 0 );
				padding.backwardsChannel(gradientPadding,batch,channel,expectedGradientInput);
			}
		}

		// sanity check to make sure it's not a bad unit test
		assertTrue(TensorOps_F64.elementSum(expectedGradientInput) != 0 );

		// check the result
		DeepUnitTest.assertEquals(expectedGradientInput, gradientInput, DeepBoofConstants.TEST_TOL_F64);
	}

	/**
	 * This isn't a real gradient calculation, just something to stress the functions internally
	 */
	private void computeGradientPadding(DSpatialPadding2D_F64 input ,
										  int b, int c, int y0, int x0, int HH, int WW,
										  Tensor_F64 dpadding ) {

		for (int y = 0; y < HH; y++) {
			int yy = y+y0;
			for (int x = 0; x < WW; x++) {
				int xx = x+x0;
				int index = dpadding.idx(yy,xx);
				dpadding.d[index] += input.get(b,c,yy,xx);
			}
		}
	}
}
