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

package deepboof.tensors;

import deepboof.Tensor;

import java.util.Arrays;

/**
 * @author Peter Abeles
 */
public class Tensor_F64 extends Tensor<Tensor_F64> {

	/**
	 * Storage for tensor data.  The tensor is stored in a row-major format.
	 */
	public double d[] = new double[0];

	public Tensor_F64( int... shape ) {
		reshape(shape);
	}

	public Tensor_F64(){}

	public double get( int ...coordinate ) {
		return d[idx(coordinate)];
	}

	public double getAtIndex( int index ) {
		return d[startIndex+ index];
	}

	@Override
	public /**/double /**/getDouble(int ...coordinate) {
		return d[idx(coordinate)];
	}

	@Override
	public Object getData() {
		return d;
	}

	@Override
	public void setData(Object data) {
		this.d = (double[])data;
	}

	@Override
	protected void innerArrayGrow(int N) {
		if( d.length < N ) {
			d = new double[N];
		}
	}

	@Override
	protected int innerArrayLength() {
		return d.length;
	}

	@Override
	public Tensor_F64 create(int... shape) {
		return new Tensor_F64(shape);
	}

	@Override
	public void zero() {
		Arrays.fill(d,startIndex,startIndex+length(),0);
	}

	@Override
	public Class getDataType() {
		return double.class;
	}
}
