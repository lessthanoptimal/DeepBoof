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

import deepboof.DeepBoofConstants;
import deepboof.DeepUnitTest;
import deepboof.misc.TensorFactory_F64;
import deepboof.tensors.Tensor_F64;
import org.junit.Test;

import java.util.List;
import java.util.Random;

import static deepboof.misc.TensorOps.WI;
import static deepboof.misc.TensorOps.WT;

/**
 * Validates numerical gradient computation.  The following configuration is considered.  h(x,w) = f(g(x,w))
 *
 * g(x,w) is the layer whose gradient is being computed numerically, but for validation purposes the
 * gradient of the whole system is computed analytically.
 *
 * g(x,w) = w[i]*w[i]**2
 * f(x) = x[i]**3
 *
 * As a bonus, the analytical gradient is also found inside the function and checked for correctness.  Mostly to make
 * sure the author knows what he is doing.
 *
 * @author Peter Abeles
 */
public class TestNumericalGradient_F64 {

	Random random = new Random(234);

	@Test
	public void compareToAnalytical() {

		int numBatch = 1;
		int M = 5;

		SimpleFunction function = new SimpleFunction();
		function.initialize(WI(M));

		NumericalGradient_F64 alg = new NumericalGradient_F64();
		alg.setFunction(function);

		for (boolean sub : new boolean[]{false, true}) {

			Tensor_F64 input = TensorFactory_F64.random(random,sub,numBatch,M);
			Tensor_F64 weights = TensorFactory_F64.random(random,sub,M);

			Tensor_F64 Gout = new Tensor_F64(numBatch,M);

			function.setParameters(WT(weights));
			function.forward(input,Gout);

			// expected gradient of H
			Tensor_F64 dXW[] = computeExpected(input,weights);

			// gradient of F
			Tensor_F64 dF = computeDF(Gout);

			// let's sanity check the built in analytic gradient for fun.  Technically not
			// needed as part of this unit test
			Tensor_F64 foundDX = TensorFactory_F64.random(random,sub,numBatch,M);
			Tensor_F64 foundDW = TensorFactory_F64.random(random,sub,M);

			function.setParameters(WT(weights));
			function.learning();
			function.backwards(input,dF,foundDX,WT(foundDW));

			DeepUnitTest.assertEquals(dXW[0],foundDX, DeepBoofConstants.TEST_TOL_F64);
			DeepUnitTest.assertEquals(dXW[1],foundDW, DeepBoofConstants.TEST_TOL_F64);

			// Now let's compare the output from numerical gradient to the expected output
			foundDX = TensorFactory_F64.random(random,sub,numBatch,M);
			foundDW = TensorFactory_F64.random(random,sub,M);

			alg.differentiate(input,WT(weights),dF,foundDX,WT(foundDW));

			DeepUnitTest.assertEquals(dXW[0],foundDX, DeepBoofConstants.TEST_TOL_F64);
			DeepUnitTest.assertEquals(dXW[1],foundDW, DeepBoofConstants.TEST_TOL_F64);
		}
	}

	public static Tensor_F64 computeDF( Tensor_F64 X ) {
		Tensor_F64 dX = X.createLike();

		int N = X.length(0);
		int M = X.length(1);

		for (int batch = 0; batch < N; batch++) {
			for (int i = 0; i < M; i++) {
				double x = X.get(batch,i);

				dX.d[dX.idx(batch,i)] = 3.0*x*x;
			}
		}

		return dX;
	}

	public static Tensor_F64[] computeExpected( Tensor_F64 X , Tensor_F64 W ) {
		Tensor_F64 dX = X.createLike();
		Tensor_F64 dW = W.createLike();

		int N = X.length(0);
		int M = X.length(1);

		for (int batch = 0; batch < N; batch++) {
			for (int i = 0; i < M; i++) {
				double x = X.get(batch,i);
				double w = W.get(i);

				double inner = x*w*w;

				dX.d[dX.idx(batch,i)] = 3.0*inner*inner*w*w;
				dW.d[dW.idx(i)] += 6*inner*inner*x*w;
			}
		}

		return new Tensor_F64[]{dX,dW};
	}

	public static class SimpleFunction extends BaseDFunction<Tensor_F64> {

		Tensor_F64 weights;

		@Override
		protected void _backwards(Tensor_F64 input, Tensor_F64 dout,
								  Tensor_F64 gradientInput, List<Tensor_F64> gradientParameters) {
			Tensor_F64 dweights = gradientParameters.get(0);

			dweights.zero();

			int N = input.length(0);
			int M = input.length(1);

			for (int batch = 0; batch < N; batch++) {
				for (int i = 0; i < M; i++) {

					double x = input.get(batch,i);
					double w = weights.get(i);

					double dout_i = dout.get(batch,i);

					gradientInput.d[gradientInput.idx(batch,i)] = w*w*dout_i;
					dweights.d[dweights.idx(i)] += 2*w*x*dout_i;
				}
			}
		}

		@Override
		public void _initialize() {
			shapeParameters.add(shapeInput.clone());
			shapeOutput = shapeInput.clone();
		}

		@Override
		public void _setParameters(List<Tensor_F64> parameters) {
			weights = parameters.get(0);
		}

		@Override
		public void _forward(Tensor_F64 input, Tensor_F64 output) {

			int N = input.length(0);
			int M = input.length(1);

			for (int batch = 0; batch < N; batch++) {
				for (int i = 0; i < M; i++) {

					double x = input.get(batch,i);
					double w = weights.get(i);

					output.d[output.idx(batch,i)] = x*w*w;
				}
			}
		}

		@Override
		public Class<Tensor_F64> getTensorType() {
			return Tensor_F64.class;
		}
	}
}