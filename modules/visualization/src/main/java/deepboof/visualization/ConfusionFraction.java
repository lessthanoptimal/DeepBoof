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

package deepboof.visualization;

import org.ejml.data.DenseMatrix64F;

import java.util.List;

/**
 * Stores a confusion matrix of results in an N by N double row-major matrix and provides functions for computing
 * different statistical properties.  To compute different statistical properties first invoke {@link #precompute()}.
 *
 * @author Peter Abeles
 */
public class ConfusionFraction {
	/**
	 * N by N matrix of counts.  Each element represents the fraction of times category 'row' was labeled
	 * 'col'.  Thus, the diagonals are the fraction of the time that each category has been correctly labeled.
	 */
	public DenseMatrix64F M;

	// precompute the sums along rows and columns
	public double sumCols[];

	/**
	 * Optional labels for each category.
	 */
	public List<String> labels;
	/**
	 * Number of categories
	 */
	public int N;

	public ConfusionFraction(List<String> labels) {
		this(labels.size());
		this.labels = labels;
	}

	public ConfusionFraction(int N) {
		this.N = N;
		M = new DenseMatrix64F(N,N);
		sumCols = new double[N];
	}

	public ConfusionFraction( ConfusionCounts counts ) {
		this(counts.N);
		this.labels = counts.labels;
		counts.precompute();
		for (int row = 0; row < N; row++) {
			double total = counts.sumRows[row];
			for (int col = 0; col < N; col++) {
				if( total == 0)
					M.unsafe_set(row,col, 0 );
				else
					M.unsafe_set(row,col, counts.get(row,col)/total );
			}
		}
		precompute();
	}

	public void precompute() {
		// rows always sum up to one, no need to precompute

		// sum up along the columns
		for (int col = 0; col < N; col++) {
			int sum = 0;
			for (int row = 0; row < N; row++) {
				sum += get(row,col);
			}
			sumCols[col] = sum;
		}
	}

	/**
	 * Returns the fraction of the time category 'row' was labeled 'col'.
	 * @param row Index of category
	 * @param col Index of category
	 * @return fraction from 0 to 1, inclusive
	 */
	public double get( int row , int col ) {
		return M.unsafe_get(row,col);
	}

	/**
	 * Returns the fraction of time it correctly labeled this category
	 */
	public double precision( int which ) {
		return get(which,which);
	}

	public double recall( int which ) {
return get(which,which)/sumCols[which];
	}
}
