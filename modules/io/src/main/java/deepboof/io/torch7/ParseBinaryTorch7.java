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

package deepboof.io.torch7;

import java.io.IOException;

/**
 * Parser for binary Torch 7 serialized objects.
 *
 * @author Peter Abeles
 */
public class ParseBinaryTorch7 extends ParseTorch7 {

	// is the file saved in a little endian format or a big endian format
	boolean littleEndian = true;

	/**
	 * Constructor which allows you to configure byte order
	 *
	 * @param littleEndian true if it was written on a system in little endian byte order or false for big endian.
	 */
	public ParseBinaryTorch7(boolean littleEndian) {
		this.littleEndian = littleEndian;
	}

	/**
	 * Default constructor.  Little endian byte order.
	 */
	public ParseBinaryTorch7() {
	}

	@Override
	public int[] readShape( int dimension ) throws IOException {
		int[] shape = new int[dimension];
		for (int i = 0; i < dimension; i++) {
			shape[i] = (int)readS64();
		}
		return shape;
	}

	@Override
	public TorchType readType() throws IOException {
		return TorchType.valueToType(readS32());
	}

	@Override
	public boolean readBoolean() throws IOException {
		int value = readS32();
		return value == 1;
	}

	@Override
	public double readDouble() throws IOException {
		long value = input.readLong();
		if( littleEndian ) {
			value = Long.reverseBytes(value);
		}
		return Double.longBitsToDouble(value);
	}

	@Override
	public float readFloat() throws IOException {
		int value = input.readInt();
		if( littleEndian ) {
			value = Integer.reverseBytes(value);
		}
		return Float.intBitsToFloat(value);
	}

	@Override
	public String readString() throws IOException {
		int length = readS32();
		byte[] array = new byte[length];

		for (int i = 0; i < length; i++) {
			array[i] = (byte)readU8();
		}

		return new String(array);
	}

	@Override
	public long readS64() throws IOException {
		if( littleEndian ) {
			return Long.reverseBytes(input.readLong());
		} else {
			return input.readLong();
		}
	}

	@Override
	public int readS32() throws IOException {
		if( littleEndian ) {
			return Integer.reverseBytes(input.readInt());
		} else {
			return input.readInt();
		}
	}

	@Override
	public int readU8() throws IOException {
		return input.readByte() & 0xFF;
	}

	@Override
	public void readArrayDouble(int size, double[] storage) throws IOException {
		// read it in one big batch to make it faster
		byte[] tmp = new byte[size*8];
		input.readFully(tmp);
		if( littleEndian ) {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 8) {
				long a =  (tmp[idx]&0xFF) | (tmp[idx+1]&0xFF)<<8 | (tmp[idx+2]&0xFF)<<16 | (long)(tmp[idx+3]&0xFF) << 24L;
				long b = (tmp[idx+4]&0xFF) | (tmp[idx+5]&0xFF)<<8 | (tmp[idx+6]&0xFF)<<16 | (long)(tmp[idx+7]&0xFF) << 24;

				storage[i] = Double.longBitsToDouble(b << 32 | a );
			}
		} else {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 8) {
				long a = (tmp[idx+3]&0xFF) | (tmp[idx+2]&0xFF)<<8 | (tmp[idx+1]&0xFF)<<16 | (long)(tmp[idx]&0xFF) << 24;
				long b = (tmp[idx+7]&0xFF) | (tmp[idx+6]&0xFF)<<8 | (tmp[idx+5]&0xFF)<<16 | (long)(tmp[idx+4]&0xFF) << 24;

				storage[i] = Double.longBitsToDouble(a << 32 | b );
			}
		}
	}

	@Override
	public void readArrayFloat(int size, float[] storage) throws IOException {
		// read it in one big batch to make it faster
		byte[] tmp = new byte[size*4];
		input.readFully(tmp);
		if( littleEndian ) {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 4) {
				int v = (tmp[idx]&0xFF) | (tmp[idx+1]&0xFF)<<8 | (tmp[idx+2]&0xFF)<<16 | (tmp[idx+3]&0xFF) << 24;
				storage[i] = Float.intBitsToFloat(v);
			}
		} else {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 4) {
				int v = (tmp[idx+3]&0xFF) | (tmp[idx+2]&0xFF)<<8 | (tmp[idx+1]&0xFF)<<16 | (tmp[idx]&0xFF) << 24;
				storage[i] = Float.intBitsToFloat(v);
			}
		}
	}

	@Override
	public void readArrayChar(int size, char[] storage) throws IOException {
		size = size/2 + size%2;
		if( littleEndian ) {
			for (int i = 0; i < size; i++) {
				storage[i] = (char)Short.reverseBytes(input.readShort());
			}
		} else {
			for (int i = 0; i < size; i++) {
				storage[i] = (char)input.readShort();
			}
		}
	}

	@Override
	public void readArrayLong(int size, long[] storage) throws IOException {
		// read it in one big batch to make it faster
		byte[] tmp = new byte[size*8];
		input.readFully(tmp);
		if( littleEndian ) {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 8) {
				long a =  (tmp[idx]&0xFF) | (tmp[idx+1]&0xFF)<<8 | (tmp[idx+2]&0xFF)<<16 | (long)(tmp[idx+3]&0xFF) << 24L;
				long b = (tmp[idx+4]&0xFF) | (tmp[idx+5]&0xFF)<<8 | (tmp[idx+6]&0xFF)<<16 | (long)(tmp[idx+7]&0xFF) << 24;

				storage[i] = b << 32 | a;
			}
		} else {
			int idx = 0;
			for (int i = 0; i < size; i++, idx += 8) {
				long a = (tmp[idx+3]&0xFF) | (tmp[idx+2]&0xFF)<<8 | (tmp[idx+1]&0xFF)<<16 | (long)(tmp[idx]&0xFF) << 24;
				long b = (tmp[idx+7]&0xFF) | (tmp[idx+6]&0xFF)<<8 | (tmp[idx+5]&0xFF)<<16 | (long)(tmp[idx+4]&0xFF) << 24;

				storage[i] = a << 32 | b;
			}
		}
	}

	@Override
	public void readArrayByte(int size, byte[] storage) throws IOException {
		input.readFully(storage,0,size);
	}
}
