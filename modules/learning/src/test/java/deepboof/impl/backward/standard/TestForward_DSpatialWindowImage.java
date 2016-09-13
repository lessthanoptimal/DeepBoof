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

import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.impl.forward.standard.BaseSpatialWindow;
import deepboof.impl.forward.standard.ChecksSpatialWindow;
import deepboof.impl.forward.standard.ConstantPadding2D_F64;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * @author Peter Abeles
 */
public class TestForward_DSpatialWindowImage extends ChecksSpatialWindow {

	public BaseSpatialWindow<Tensor_F64,ConstantPadding2D_F64> create(ConfigSpatial config ) {
		return (BaseSpatialWindow)new Helper(config,createPadding());
	}

	private DConstantPadding2D_F64 createPadding() {
		ConfigPadding config = new ConfigPadding();
		config.y0 = pad;
		config.x0 = pad;
		config.y1 = pad;
		config.x1 = pad;

		return new DConstantPadding2D_F64(config);
	}

	public class Helper extends DSpatialWindowImage<Tensor_F64,DConstantPadding2D_F64> {

		public Helper(ConfigSpatial configSpatial, DConstantPadding2D_F64 padding ) {
			super(configSpatial, padding);
		}

		@Override
		protected void forwardAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {
			for (int c = 0; c < C; c++) {
				double sum = 0;
				for (int y = 0; y < HH; y++) {
					for (int x = 0; x < WW; x++) {
						sum += input.get(batch,c,y+inY,x+inX);
					}
				}

				output.d[output.idx(batch,c,outY,outX)] = sum;
			}
		}

		@Override
		protected void forwardAt_border(DConstantPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
			for (int c = 0; c < C; c++) {
				double sum = 0;
				for (int y = 0; y < HH; y++) {
					for (int x = 0; x < WW; x++) {
						sum += padded.get(batch,c,y+padY,x+padX);
					}
				}

				output.d[output.idx(batch,c,outY,outX)] = sum;
			}
		}

		@Override
		protected void _backwards(Tensor_F64 input, Tensor_F64 dout, Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {

		}

		@Override
		protected void backwardsAt_inner(Tensor_F64 input, int batch, int inY, int inX, int outY, int outX) {

		}

		@Override
		protected void backwardsAt_border(DConstantPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {

		}

		@Override
		public Class<Tensor_F64> getTensorType() {
			return Tensor_F64.class;
		}

		@Override
		public void _setParameters(List<Tensor_F64> parameters) {}

		@Override
		public void _forward(Tensor_F64 input, Tensor_F64 output) {
			forwardImage(input, output);
		}
	}

}