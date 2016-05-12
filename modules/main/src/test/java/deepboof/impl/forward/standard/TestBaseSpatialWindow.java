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

import deepboof.DeepUnitTest;
import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.forward.SpatialPadding2D_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.List;

import static deepboof.misc.TensorOps.WI;
import static org.junit.Assert.*;

/**
 * @author Peter Abeles
 */
public class TestBaseSpatialWindow {

	@Test
	public void init() {
		ConfigSpatial configSpatial = new ConfigSpatial();
		ConfigPadding configPadding = new ConfigPadding();

		configSpatial.HH = 3;
		configSpatial.WW = 4;
		configSpatial.periodY = 5;
		configSpatial.periodX = 6;

		configPadding.x0 = 1;
		configPadding.y0 = 2;
		configPadding.x1 = 3;
		configPadding.y1 = 4;

		Helper alg = new Helper(configSpatial,new ConstantPadding2D_F64(configPadding));

		int C = 2;
		int H = 10;
		int W = 13;

		alg.initialize(C,H,W);

		assertEquals(C,alg.C);
		assertEquals(H,alg.H);
		assertEquals(W,alg.W);

		assertEquals(3,alg.HH);
		assertEquals(4,alg.WW);

		assertEquals((2+4)+H,alg.Hp);
		assertEquals((1+3)+W,alg.Wp);

		assertEquals(1+(alg.Hp-3)/5,alg.Ho);
		assertEquals(1+(alg.Wp-4)/6,alg.Wo);

		DeepUnitTest.assertEquals(WI(C,alg.Ho,alg.Wo),alg.getOutputShape());
	}

	@Test
	public void isEntirelyBorder() {
		ConfigSpatial configSpatial = new ConfigSpatial();
		ConfigPadding configPadding = new ConfigPadding();

		configSpatial.HH = 2;
		configSpatial.WW = 2;
		configSpatial.periodY = 1;
		configSpatial.periodX = 1;

		configPadding.y0 = 3;
		configPadding.x0 = 2;
		configPadding.y1 = 2;
		configPadding.x1 = 2;


		Helper alg = new Helper(configSpatial,new ConstantPadding2D_F64(configPadding));

		alg.initialize(2,12,13);


		assertFalse(alg.isEntirelyBorder(0, 0));
		assertFalse(alg.isEntirelyBorder(0, alg.Wo-3));
		assertFalse(alg.isEntirelyBorder(alg.Ho-3, 0));

		assertTrue(alg.isEntirelyBorder(0,  alg.Wo));
		assertTrue(alg.isEntirelyBorder(alg.Ho, 0));
		assertTrue(alg.isEntirelyBorder(0,  alg.Wo-1));
		assertTrue(alg.isEntirelyBorder(alg.Ho-1, 0));
	}

	@Test
	public void innerLowerExtent() {

		assertEquals(0,BaseSpatialWindow.innerLowerExtent(1,0));
		assertEquals(0,BaseSpatialWindow.innerLowerExtent(3,0));
		assertEquals(1,BaseSpatialWindow.innerLowerExtent(1,1));
		assertEquals(3,BaseSpatialWindow.innerLowerExtent(1,3));

		assertEquals(0,BaseSpatialWindow.innerLowerExtent(2,0));
		assertEquals(1,BaseSpatialWindow.innerLowerExtent(2,1));
		assertEquals(1,BaseSpatialWindow.innerLowerExtent(2,2));
		assertEquals(2,BaseSpatialWindow.innerLowerExtent(2,3));
		assertEquals(0,BaseSpatialWindow.innerLowerExtent(2,0));
		assertEquals(1,BaseSpatialWindow.innerLowerExtent(2,1));
		assertEquals(1,BaseSpatialWindow.innerLowerExtent(2,2));
		assertEquals(2,BaseSpatialWindow.innerLowerExtent(2,3));
	}

	@Test
	public void innerUpperExtent() {
		assertEquals(9,BaseSpatialWindow.innerUpperExtent(1,1,0,10));
		assertEquals(4,BaseSpatialWindow.innerUpperExtent(1,2,0,10));
		assertEquals(8,BaseSpatialWindow.innerUpperExtent(2,1,0,10));
		assertEquals(7,BaseSpatialWindow.innerUpperExtent(3,1,0,10));
		assertEquals(3,BaseSpatialWindow.innerUpperExtent(1,3,0,10));
		assertEquals(2,BaseSpatialWindow.innerUpperExtent(2,3,0,10));

		assertEquals(11,BaseSpatialWindow.innerUpperExtent(1,1,2,10));
		assertEquals(5 ,BaseSpatialWindow.innerUpperExtent(1,2,2,10));
		assertEquals(10,BaseSpatialWindow.innerUpperExtent(2,1,2,10));
		assertEquals(9 ,BaseSpatialWindow.innerUpperExtent(3,1,2,10));
		assertEquals(3 ,BaseSpatialWindow.innerUpperExtent(1,3,2,10));
		assertEquals(3 ,BaseSpatialWindow.innerUpperExtent(2,3,2,10));
		assertEquals(3 ,BaseSpatialWindow.innerUpperExtent(2,3,1,10));
		assertEquals(2 ,BaseSpatialWindow.innerUpperExtent(3,3,1,10));
		assertEquals(3 ,BaseSpatialWindow.innerUpperExtent(2,3,2,10));
	}

	public static class Helper extends BaseSpatialWindow<Tensor_F64,SpatialPadding2D_F64> {

		public Helper(ConfigSpatial config, SpatialPadding2D_F64 padding) {
			super(config, padding);
		}

		@Override
		public void _setParameters(List<Tensor_F64> parameters) {}

		@Override
		public void _forward(Tensor_F64 input, Tensor_F64 output) {

		}

		@Override
		public Class<Tensor_F64> getTensorType() {
			return Tensor_F64.class;
		}
	}
}