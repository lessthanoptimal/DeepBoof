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
import deepboof.Function;
import deepboof.Tensor;
import deepboof.factory.FactoryBackwards;
import deepboof.misc.TensorFactory;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.fail;

/**
 * @author Peter Abeles
 */
public abstract class CheckDerivativePadding<T extends Tensor<T>,P extends DSpatialPadding2D<T>>
{
	protected Random random = new Random(234);

	protected TensorFactory<T> tensorFactory;
	protected FactoryBackwards<T> factoryD;

	protected Accuracy tolerance = Accuracy.RELAXED_A;

	protected P alg;

	public abstract P createBackwards();

	@Before
	public void before() {
		tensorFactory = new TensorFactory<>(createBackwards().getTensorType());
		factoryD = new FactoryBackwards<>(createBackwards().getTensorType());

		alg = createBackwards();
	}

	/**
	 * If it's not a 2D tensor it should throw an exception
	 */
	@Test
	public void sanityCheckPaddedShape_channel() {
		int inputShape[] = new int[]{3,4,10,12};

		int padRow = alg.getPaddingRow0()+alg.getPaddingRow1();
		int padCol = alg.getPaddingCol0()+alg.getPaddingCol1();

		int paddedSpatial[] = new int[]{ inputShape[0],inputShape[1],inputShape[2]+padRow, inputShape[3]+padCol};


		T dpadded = tensorFactory.random(random, false , paddedSpatial);
		T foundDInput = tensorFactory.random(random, false , paddedSpatial);

		sanityCheckPaddedShapeChannel(dpadded,foundDInput);
		sanityCheckPaddedShapeChannel( tensorFactory.random(random, false,
				inputShape[2]+padRow+1, inputShape[3]+padCol) ,foundDInput);
		sanityCheckPaddedShapeChannel( tensorFactory.random(random, false,
				inputShape[2]+padRow, inputShape[3]+padCol+1) ,foundDInput);
	}

	private void sanityCheckPaddedShapeChannel( T dpadded, T foundDInput ) {
		try {
			alg.backwardsChannel(dpadded, 0, 0, foundDInput);
			fail("Exception should have been thrown");
		} catch( RuntimeException e) {}
	}

	/**
	 * If it's not a 3D tensor it should throw an exception
	 */
	@Test
	public void sanityCheckPaddedShape_image() {
		int inputShape[] = new int[]{3,4,10,12};

		int padRow = alg.getPaddingRow0()+alg.getPaddingRow1();
		int padCol = alg.getPaddingCol0()+alg.getPaddingCol1();

		int paddedSpatial[] = new int[]{ inputShape[0],inputShape[1],inputShape[2]+padRow, inputShape[3]+padCol};


		T dpadded = tensorFactory.random(random, false , paddedSpatial);
		T foundDInput = tensorFactory.random(random, false , paddedSpatial);

		checkFailPaddedShapeImage(dpadded,foundDInput);
		checkFailPaddedShapeImage( tensorFactory.random(random, false,
				inputShape[1],inputShape[2]+padRow+1, inputShape[3]+padCol) ,foundDInput);
		checkFailPaddedShapeImage( tensorFactory.random(random, false,
				inputShape[1],inputShape[2]+padRow, inputShape[3]+padCol+1) ,foundDInput);
		checkFailPaddedShapeImage( tensorFactory.random(random, false,
				inputShape[1]+1,inputShape[2]+padRow, inputShape[3]+padCol) ,foundDInput);
	}

	private void checkFailPaddedShapeImage( T dpadded, T foundDInput ) {
		try {
			alg.backwardsImage(dpadded, 0, foundDInput);
			fail("Exception should have been thrown");
		} catch( RuntimeException e) {}
	}

	/**
	 * Tests the {@link DSpatialPadding2D#backwardsChannel(Tensor, int, int, Tensor)}
	 */
	@Test
	public void checkBackwardsRandomInput_channel() {

		int inputShape[] = new int[]{3,4,10,12};

		int padRow = alg.getPaddingRow0()+alg.getPaddingRow1();
		int padCol = alg.getPaddingCol0()+alg.getPaddingCol1();

		int paddedImage[] = new int[]{ inputShape[2]+padRow, inputShape[3]+padCol};
		int paddedSpatial[] = new int[]{ inputShape[0],inputShape[1],inputShape[2]+padRow, inputShape[3]+padCol};

		NumericalGradient<T> numeric = factoryD.createNumericalGradient();
		numeric.setFunction(new PaddingFunction(alg,inputShape));


		List<T> emptyList = new ArrayList<>();

		for (boolean sub : new boolean[]{false, true}) {
			T inputTensor = tensorFactory.random(random, sub, inputShape);

			// storage for the found input tensor gradient
			T foundDInput = tensorFactory.random(random, sub, inputShape);

			// random gradient of padded input tensor
			T dpaddedSpatial = tensorFactory.random(random, sub, paddedSpatial);

			// numerically computed input tensor gradient
			T expectedSpatial = tensorFactory.random(random, sub, inputShape);

			// compute ground truth
			numeric.differentiate(inputTensor,emptyList,dpaddedSpatial,expectedSpatial,emptyList);

			// compute the gradient one channel at a time
			for (int batch = 0; batch < inputShape[0]; batch++) {
				for (int channel = 0; channel < inputShape[1]; channel++) {

					int index = dpaddedSpatial.idx(batch,channel,0,0);
					T dpadded = dpaddedSpatial.subtensor(index,paddedImage);

					alg.backwardsChannel(dpadded,batch,channel,foundDInput);
				}
			}

			// compare results
			DeepUnitTest.assertEquals(expectedSpatial,foundDInput, tolerance );
		}
	}

	/**
	 * Tests the {@link DSpatialPadding2D#backwardsImage(Tensor, int, Tensor)}}
	 */
	@Test
	public void checkBackwardsRandomInput_image() {

		int inputShape[] = new int[]{3,4,10,12};

		int padRow = alg.getPaddingRow0()+alg.getPaddingRow1();
		int padCol = alg.getPaddingCol0()+alg.getPaddingCol1();

		int paddedImage[] = new int[]{ inputShape[1], inputShape[2]+padRow, inputShape[3]+padCol};
		int paddedSpatial[] = new int[]{ inputShape[0],inputShape[1],inputShape[2]+padRow, inputShape[3]+padCol};

		NumericalGradient<T> numeric = factoryD.createNumericalGradient();
		numeric.setFunction(new PaddingFunction(alg,inputShape));


		List<T> emptyList = new ArrayList<>();

		for (boolean sub : new boolean[]{false, true}) {
			T inputTensor = tensorFactory.random(random, sub, inputShape);

			// storage for the found input tensor gradient
			T foundDInput = tensorFactory.random(random, sub, inputShape);

			// random gradient of padded input tensor
			T dpaddedSpatial = tensorFactory.random(random, sub, paddedSpatial);

			// numerically computed input tensor gradient
			T expectedSpatial = tensorFactory.random(random, sub, inputShape);

			// compute ground truth
			numeric.differentiate(inputTensor,emptyList,dpaddedSpatial,expectedSpatial,emptyList);

			// compute the gradient one channel at a time
			for (int batch = 0; batch < inputShape[0]; batch++)
			{
				int index = dpaddedSpatial.idx(batch,0,0,0);
				T dpadded = dpaddedSpatial.subtensor(index,paddedImage);

				alg.backwardsImage(dpadded,batch,foundDInput);
			}

			// compare results
			DeepUnitTest.assertEquals(expectedSpatial,foundDInput, tolerance );
		}
	}

	protected abstract void applyPadding( T input , T output );

	public class PaddingFunction implements Function<T> {
		DSpatialPadding2D<T> padding;
		int inputShape[];

		public PaddingFunction(DSpatialPadding2D<T> padding, int inputShape[]) {
			this.padding = padding;
			this.inputShape = inputShape;
		}

		@Override
		public void initialize(int... shapeInput) {this.inputShape=shapeInput;}

		@Override
		public void setParameters(List<T> parameters) {}

		@Override
		public List<T> getParameters() {return null;}

		@Override
		public void forward(T input, T output) {
			applyPadding(input, output);
		}

		@Override
		public List<int[]> getParameterShapes() {
			return new ArrayList<>();
		}

		@Override
		public int[] getOutputShape() {

			int shape[] = padding.shapeGivenInput(inputShape);
			return new int[]{shape[1],shape[2],shape[3]};
		}

		@Override
		public Class<T> getTensorType() {
			return padding.getTensorType();
		}
	}
}
