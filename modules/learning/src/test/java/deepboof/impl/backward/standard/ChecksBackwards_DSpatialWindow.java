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

package deepboof.impl.backward.standard;

import deepboof.DFunction;
import deepboof.DeepBoofConstants;
import deepboof.backward.DSpatialPadding2D;
import deepboof.backward.DSpatialPadding2D_F64;
import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 *
 *
 * @author Peter Abeles
 */
public abstract class ChecksBackwards_DSpatialWindow {
	Random rand = new Random(234);

	final int pad = 2;

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

	@Test
	public void entirelyInside() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 original = TensorFactory_F64.random(rand, sub,N,C,2,2);

			configSpatial = new ConfigSpatial();
			configSpatial.WW = 3;
			configSpatial.HH = 3;

			DFunction<Tensor_F64> helper = create(configSpatial, createPadding());

			helper.initialize(C,6,8);

			Tensor_F64 gradientInput = new Tensor_F64(WI(N,C,6,8));
			Tensor_F64 dout = new Tensor_F64(WI(N,helper.getOutputShape()));

			List<Tensor_F64> gradientParameters = new ArrayList<>();

			helper.backwards(original, dout, gradientInput, gradientParameters);

			compareToBruteForce(original, dout, gradientInput);
		}
	}

	@Test
	public void insideAndOutside() {
		for (boolean sub : new boolean[]{false, true}) {

		}
	}

	protected void compareToBruteForce(Tensor_F64 input , Tensor_F64 dout, Tensor_F64 gradientInput ) {

		DSpatialPadding2D_F64 padding = createPadding();

		int periodY = configSpatial.periodY;
		int periodX = configSpatial.periodX;
		int HH = configSpatial.HH;
		int WW = configSpatial.WW;

		int H = input.length(2)+pad*2;
		int W = input.length(3)+pad*2;

		for (int batch = 0; batch < N; batch++) {
			for (int channel = 0; channel < C; channel++) {
				int outY = 0;
				for (int y = 0; y <= H-HH; y += periodY, outY++) {
					int outX = 0;
					for (int x = 0; x <= W-WW; x += periodX, outX++) {
						double expected = sumWindow(padding,batch,channel, y, x, HH, WW);
						expected *= dout.get(batch,channel,y,x);
						double foundValue = gradientInput.get(batch,channel,outY,outX);
						assertEquals(y+" "+x,expected,foundValue, DeepBoofConstants.TEST_TOL_F64);
					}
				}
			}
		}
	}

	private double sumWindow(DSpatialPadding2D_F64 input , int b, int c, int y0, int x0, int HH, int WW) {
		double sum = 0;

		for (int y = 0; y < HH; y++) {
			int yy = y+y0;
			for (int x = 0; x < WW; x++) {
				int xx = x+x0;
				sum += input.get(b,c,yy,xx);
			}
		}

		return sum;
	}
}
