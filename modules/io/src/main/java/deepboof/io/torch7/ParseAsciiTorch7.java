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
 * @author Peter Abeles
 */
public class ParseAsciiTorch7 extends ParseTorch7 {
	byte buffer[] = new byte[1024];

	@Override
	public int[] readShape(int dimension) throws IOException {
		String line = readInnerString();
		String words[] = line.split(" ");
		if( words.length != dimension )
			throw new IOException("Unexpected number of words");
		int[] shape = new int[ dimension ];
		for (int i = 0; i < dimension; i++) {
			shape[i] = Integer.parseInt(words[i]);
		}
		return shape;
	}

	@Override
	public TorchType readType() throws IOException {
		return TorchType.valueToType(readS32());
	}

	@Override
	public boolean readBoolean() throws IOException {
		return readS32() != 0;
	}

	@Override
	public double readDouble() throws IOException {
		return Double.parseDouble(readInnerString());
	}

	@Override
	public float readFloat() throws IOException {
		return Float.parseFloat(readInnerString());
	}

	@Override
	public String readString() throws IOException {
		int length = Integer.parseInt(readInnerString());

		if( length > buffer.length )
			throw new IOException("Need to increase size of buffer to read this string");

		input.readFully(buffer,0,length+1);
		if( buffer[length] != 0x0A )
			throw new IOException("Unexpected string ending");
		return new String(buffer,0,length);
	}

	@Override
	public long readS64() throws IOException {
		return Integer.parseInt(readInnerString());
	}

	@Override
	public int readS32() throws IOException {
		return Integer.parseInt(readInnerString());
	}

	@Override
	public int readU8() throws IOException {
		return Integer.parseInt(readInnerString());
	}

	@Override
	public void readArrayDouble(int size, double[] storage) throws IOException {
		String line = readInnerString();
		String words[] = line.split(" ");
		if( words.length != size )
			throw new IOException("Unexpected number of words "+size+" found "+words.length);
		for (int i = 0; i < size; i++) {
			if( words[i].endsWith("nan"))
				storage[i] = Double.NaN;
			else
				storage[i] = Double.parseDouble(words[i]);
		}
	}

	@Override
	public void readArrayFloat(int size, float[] storage) throws IOException {
		String line = readInnerString();
		String words[] = line.split(" ");
		if( words.length != size )
			throw new IOException("Unexpected number of words "+size+" found "+words.length);
		for (int i = 0; i < size; i++) {
			storage[i] = Float.parseFloat(words[i]);
		}
	}

	@Override
	public void readArrayChar(int size, char[] storage) throws IOException {
		for (int i = 0; i < size/2; i++) {
			storage[i] = (char)Short.reverseBytes(input.readShort());
		}
		if( size%2 == 1 ) {
			storage[size/2] = (char)input.readByte();
		}
		input.readByte();
	}

	@Override
	public void readArrayByte(int size, byte[] storage) throws IOException {
		input.readFully(storage,0,size);
		input.readByte();
	}

	@Override
	public void readArrayLong(int size, long[] storage) throws IOException {
		String line = readInnerString();
		String words[] = line.split(" ");
		if( words.length != size )
			throw new IOException("Unexpected number of words "+size+" found "+words.length);
		for (int i = 0; i < size; i++) {
			storage[i] = Long.parseLong(words[i]);
		}
	}

	private String readInnerString() throws IOException {
		int length = 0;
		while( true ) {
			int value = input.readUnsignedByte();
			if( value == 0x0A ) {
				break;
			}
			buffer[length++] = (byte)value;

			if( buffer.length == length ) {
				growBuffer();
			}
		}
		return new String(buffer,0,length);
	}

	private void growBuffer() {
		byte tmp[] = new byte[ buffer.length + 1024];
		System.arraycopy(buffer,0,tmp,0,buffer.length);
		buffer = tmp;
	}
}
