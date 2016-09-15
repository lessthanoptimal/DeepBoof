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

import deepboof.Accuracy;
import deepboof.DeepUnitTest;
import deepboof.Tensor;
import deepboof.forward.ChecksForward;
import org.junit.Test;

import java.util.Arrays;

/**
 * @author Peter Abeles
 */
public abstract class ChecksForward_DBatchNorm<T extends Tensor<T>> extends ChecksForward<T> {

	protected boolean gammaBeta;

	double tolerance;

    public ChecksForward_DBatchNorm( double tolerance ) {
        super(2);
        this.tolerance = tolerance;
    }

	public abstract DBatchNorm<T> createForwards( boolean gammaBeta );

	   /**
     * Test it's behavior when in evaluation mode
     */
    @Test
	public void forwards_evaluate() {
		for (int config = 0; config < numberOfConfigurations; config++) {
			for (Case test : createTestInputs()) {
				DBatchNorm<T> alg = createForwards(config);
				alg.initialize(test.inputShape);

				if (alg.hasGammaBeta()) {
					int pshape[] = alg.getParameterShapes().get(0);
					T params = tensorFactory.random(random, false, pshape);
					alg.setParameters(Arrays.asList(params));
				}

				// compute the the statistics from one set of data
				T input = tensorFactory.randomM(random, false, test.minibatch, test.inputShape);
				T output = tensorFactory.randomM(random, false, test.minibatch, test.inputShape);

				alg.learning();
				alg.forward(input, output);

				T origInput = input.copy();
				T origOutput = output.copy();

				T origMean = alg.getMean(null);
				T origVar = alg.getVariance(null);

				// put it into evaluation mode and provide it new data.  The statistics should
				// remain the same
				input = tensorFactory.randomM(random, false, test.minibatch, test.inputShape);
				alg.evaluating();
				alg.forward(input, output);

				DeepUnitTest.assertEquals(origMean, alg.getMean(null), Accuracy.STANDARD);
				DeepUnitTest.assertEquals(origVar, alg.getVariance(null), Accuracy.STANDARD);

				// output should also have changed
				DeepUnitTest.assertNotEquals(origOutput, output, Accuracy.STANDARD);

				// now give it the original input and see if it produces the original output
				alg.forward(origInput, output);
				DeepUnitTest.assertEquals(origOutput, output, Accuracy.STANDARD);
			}
		}
    }
}
