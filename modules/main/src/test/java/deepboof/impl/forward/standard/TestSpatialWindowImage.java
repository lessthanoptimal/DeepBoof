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

import deepboof.forward.ConfigPadding;
import deepboof.forward.ConfigSpatial;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * @author Peter Abeles
 */
public class TestSpatialWindowImage extends ChecksSpatialWindow {

	@Override
	public BaseSpatialWindow<Tensor_F64, ConstantPadding2D_F64> create(ConfigSpatial config) {
		return new Helper(config);
	}

	public class Helper extends SpatialWindowImage<Tensor_F64,ConstantPadding2D_F64>
	{

		public Helper(ConfigSpatial configSpatial) {
			super(configSpatial, null);

			ConfigPadding config = new ConfigPadding();
			config.y0 = pad;
			config.x0 = pad;
			config.y1 = pad;
			config.x1 = pad;

			this.padding =  new ConstantPadding2D_F64(config);
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
		protected void forwardAt_border(ConstantPadding2D_F64 padded, int batch, int padY, int padX, int outY, int outX) {
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
		public void _forward(Tensor_F64 input, Tensor_F64 output) {
			forwardImage(input, output);
		}

		@Override
		public Class<Tensor_F64> getTensorType() {
			return Tensor_F64.class;
		}

		@Override
		public void _setParameters(List<Tensor_F64> parameters) {

		}
	}
}
