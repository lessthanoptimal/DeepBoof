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
import deepboof.tensors.Tensor_F64;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public class TestBaseSpatialPadding2D {

	ConfigPadding config;

	@Before
	public void before() {
		config = new ConfigPadding();
		config.y0 = 2;
		config.x0 = 1;
		config.y1 = 4;
		config.x1 = 3;
	}

	@Test
	public void setInput() {
		Helper alg = new Helper(config);

		alg.setInput(new Tensor_F64(3,4,6,7));

		assertEquals(2,alg.ROW0);
		assertEquals(1,alg.COL0);

		assertEquals(8,alg.ROW1);
		assertEquals(8,alg.COL1);

	}

	@Test
	public void getPaddingXXXX() {
		Helper alg = new Helper(config);
		assertEquals(config.x0,alg.getPaddingCol0());
		assertEquals(config.y0,alg.getPaddingRow0());
		assertEquals(config.x1,alg.getPaddingCol1());
		assertEquals(config.y1,alg.getPaddingRow1());
	}

	@Test
	public void shapeGivenInput() {
		Helper alg = new Helper(config);

		int found[] = alg.shapeGivenInput(new int[]{2,9,10});
		DeepUnitTest.assertEquals(new int[]{2,15,14},found);

		found = alg.shapeGivenInput(new int[]{5,2,9,10});
		DeepUnitTest.assertEquals(new int[]{5,2,15,14},found);
	}

	private static class Helper extends BaseSpatialPadding2D<Tensor_F64> {

		public Helper(ConfigPadding config) {
			super(config);
		}

		@Override
		public Class getDataType() {
			return double.class;
		}

		@Override
		public Class<Tensor_F64> getTensorType() {
			return Tensor_F64.class;
		}
	}
}