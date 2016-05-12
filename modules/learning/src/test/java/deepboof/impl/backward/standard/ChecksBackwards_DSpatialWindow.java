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

import deepboof.forward.ConfigSpatial;
import deepboof.impl.forward.standard.BaseSpatialWindow;
import deepboof.impl.forward.standard.ConstantPadding2D_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.Random;

/**
 * @author Peter Abeles
 */
public abstract class ChecksBackwards_DSpatialWindow {
		Random rand = new Random(234);

	final int pad = 2;

	int N = 3;
	int C = 4;
	ConfigSpatial configSpatial;

	public abstract BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> create(ConfigSpatial config );

	@Test
	public void entirelyInside() {

		for (boolean sub : new boolean[]{false, true}) {

		}
	}

	@Test
	public void insideAndOutside() {
		for (boolean sub : new boolean[]{false, true}) {

		}
	}
}
