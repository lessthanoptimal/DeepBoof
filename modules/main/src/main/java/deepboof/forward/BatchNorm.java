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

/**
 * <p>Batch Normalization [1] determines the mean and standard deviation (stdev) of each input element individually
 * using the training data.  It then applies a transform (minus mean, divide stdev) for each individual
 * element to ensure it has zero mean and a standard deviation of 1 across the training set.  This alleviates
 * many problems with choosing appropriate initial parameters for inputs across all layers.</p>
 *
 * <p>During training, batch norm computes a mean and variance for each the input element.  Mean/stdev are computed by
 * finding the mean and stdev for a mini-batch and then applying a decaying average.  For
 * evaluation the previously computed mean and stdev are fixed and applied to each input element in an
 * element-wise fashion, see below.  The final mean and stdev can be computed from the decaying mean/stdev or
 * from the true mean/stdev across the entire dataset, implementation dependent.</p>
 *
 * <p>It can optionally also learn two parameters, gamma and beta, which can be used to learn to undo batch
 * normalization if helpful.  The complete transformation is shown below</p>
 *
 * <pre>
 * output[i] = ((x[i]-mean[i])/sqrt(variance[i]+EPS)*gamma[i] + beta[i]
 * </pre>
 * Where 'i' is an element in the tensor. EPS is a small number used to prevent divide by zero errors
 * and is a tuning hyper parameter.  EPS is 1e-9 for double and 1e-5 for float by default.
 *
 * <p>Training Update:</p>
 * <pre>
 * mean[i+1]  = learn_rate*mean  + (1.0-learn_rate)*mean[i]
 * stdev[i+1] = learn_rate*stdev + (1.0-learn_rate)*stdev[i] TODO change to variance?
 * </pre>
 * where (mean,stdev) with no index refers to the statistics from the current mini-batch its being trained on.
 * learn_rate determines how quickly it adjusts the mean and can have a value from 0 to 1, higher values for faster
 * but less stable learning, e.g. 0 = no learning and 1 = old results discarded.</p>
 *
 * Notes:
 * <ul>
 * <li>The shape of the output will be same as the shape of the input.</li>
 * <li>There are other variants for specific situations, e.g. {@link SpatialBatchNorm}</li>
 * </ul>
 *
 * <p>[1] Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
 * Internal Covariate Shift" 11 Feb 2015, http://arxiv.org/abs/1502.03167</p>
 *
 * @author Peter Abeles
 */
public interface BatchNorm {
    /**
     * If it returns true then it expects a second set of parameters that defines gamma and beta.
     * @return true if gamma and beta is returned.
     */
    boolean hasGammaBeta();

    double getEPS();

    /**
     * Used to specify the EPS value.  Must be invoked before setParameters()
     *
     * @param EPS Value of EPS
     */
    void setEPS(double EPS);
}
