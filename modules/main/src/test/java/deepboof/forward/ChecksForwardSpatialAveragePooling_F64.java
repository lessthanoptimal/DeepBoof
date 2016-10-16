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

package deepboof.forward;

import deepboof.misc.TensorOps;
import deepboof.tensors.Tensor_F64;

import java.util.List;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForwardSpatialAveragePooling_F64 extends ChecksForwardSpatialPooling_F64 {

	@Override
	protected double[] computeExpected(Tensor_F64 input, List<Tensor_F64> parameters, int batch , int y, int x) {

		int[] bounds = new int[]{y,x,y+config.HH,x+config.WW};
		TensorOps.boundSpatial(bounds,input.length(2),input.length(3));

		int y0 = bounds[0];
		int y1 = bounds[2];
		int x0 = bounds[1];
		int x1 = bounds[3];

		int C = input.length(1);
		double output[] = new double[C];
		for (int channel = 0; channel < C; channel++) {
			double sum = 0;

			for (int i = y0; i < y1; i++) {
				for (int j = x0; j < x1; j++) {
					sum += input.get(batch,channel,i,j);
				}
			}

			output[channel] = sum/((y1-y0)*(x1-x0));
		}

		return output;
	}


}