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

import deepboof.DeepBoofConstants;
import deepboof.forward.ConfigSpatial;
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.Random;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public abstract class ChecksSpatialWindow {
	protected Random rand = new Random(234);

	protected final int pad = 2;

	protected int N = 3;
	protected int C = 4;
	ConfigSpatial configSpatial;

	public abstract BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> create(ConfigSpatial config );

	@Test
	public void entirelyInside() {

		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 original = TensorFactory_F64.random(rand, sub,N,C,2,2);

			configSpatial = new ConfigSpatial();
			configSpatial.WW = 3;
			configSpatial.HH = 3;

			BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> helper = create(configSpatial);

			helper.initialize(C,2,2);

			Tensor_F64 found = new Tensor_F64(WI(N,C,4,4));

			helper.forward(original, found);

			compareToBruteForce(original, found);
		}
	}

	@Test
	public void insideAndOutside() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 original = TensorFactory_F64.random(rand, sub,N,C,8,9);

			configSpatial = new ConfigSpatial();
			configSpatial.HH = 3;
			configSpatial.WW = 3;

			BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> helper = create(configSpatial);

			helper.initialize(C,8,9);

			Tensor_F64 found = new Tensor_F64(WI(N,C,10,11));

			helper.forward(original,found);

			compareToBruteForce(original,found);
		}
	}

	/**
	 * Adjust the period settings and see if it responds correctly
	 */
	@Test
	public void period() {
		for( boolean sub : new boolean[]{false,true}) {
			Tensor_F64 original = TensorFactory_F64.random(rand, sub,N,C,8,9);

			configSpatial = new ConfigSpatial();
			configSpatial.periodY=3;
			configSpatial.periodX=2;
			configSpatial.HH = 3;
			configSpatial.WW = 3;

			BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> helper = create(configSpatial);

			helper.initialize(C,8,9);

			Tensor_F64 found = new Tensor_F64(WI(N,C,4,6));

			helper.forward(original, found);

			compareToBruteForce(original,found);
		}
	}

	protected void compareToBruteForce(Tensor_F64 input , Tensor_F64 found ) {

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
						double expected = sumWindow(input,batch,channel, y, x, HH, WW);
						double foundValue = found.get(batch,channel,outY,outX);
						assertEquals(y+" "+x,expected,foundValue, DeepBoofConstants.TEST_TOL_F64);
					}
				}
			}
		}
	}

	private double sumWindow(Tensor_F64 input, int b, int c, int y0, int x0, int HH, int WW) {
		int H = input.length(2)+pad*2;
		int W = input.length(3)+pad*2;

		double sum = 0;

		for (int y = 0; y < HH; y++) {
			int yy = y+y0;
			for (int x = 0; x < WW; x++) {
				int xx = x+x0;

				// border is all zero
				if( xx >= pad && xx < W-pad && yy >= pad && yy < H-pad ) {
					sum += input.get(b,c,yy-pad,xx-pad);
				}
			}
		}

		return sum;
	}
}
