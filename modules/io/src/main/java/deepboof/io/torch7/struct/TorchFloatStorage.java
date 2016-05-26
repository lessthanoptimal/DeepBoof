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

package deepboof.io.torch7.struct;

/**
 * @author Peter Abeles
 */
public class TorchFloatStorage extends TorchStorage {
	public float data[];

	public TorchFloatStorage( int size ) {
		this.data = new float[size];
		this.torchName = "torch.FloatStorage";
	}

	@Override
	public Object getDataObject() {
		return data;
	}

	@Override
	public int size() {
		return data.length;
	}

	@Override
	public void printSummary() {
		System.out.println("    storage length = " + size());
		if( size() >= 4 ) {
			int w = Math.min(3,size()/2-1);
			for (int i = 0; i < w; i++) {
				System.out.println("[ " + i + " ] " + data[i]);
			}
			System.out.println("....");

			for (int i = 0; i < w; i++) {
				int n = size() - w + i;
				System.out.println("[ " + n + " ] " + data[n]);
			}
		} else {
			for (int i = 0; i < size(); i++) {
				System.out.println("[ " + i + " ] " + data[i]);
			}
		}
	}
}
