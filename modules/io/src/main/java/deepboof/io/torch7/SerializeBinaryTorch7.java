package deepboof.io.torch7;

import java.io.IOException;

/**
 * Binary serialization of torch objects
 */
public class SerializeBinaryTorch7 extends SerializeTorch7 {

	// is the file saved in a little endian format or a big endian format
	boolean littleEndian = true;

	/**
	 * Constructor which allows you to configure byte order
	 *
	 * @param littleEndian true if it was written on a system in little endian byte order or false for big endian.
	 */
	public SerializeBinaryTorch7(boolean littleEndian) {
		this.littleEndian = littleEndian;
	}

	@Override
	public void writeShape(int[] shape) throws IOException {
		for (int i = 0; i < shape.length; i++) {
			writeS64(shape[i]);
		}
	}

	@Override
	public void writeType(TorchType type) throws IOException {
		writeS32(type.value);
	}

	@Override
	public void writeBoolean(boolean value) throws IOException {
		if( value ) {
			writeS32(1);
		} else {
			writeS32(0);
		}
	}

	@Override
	public void writeDouble(double value) throws IOException {
		long a = Double.doubleToLongBits(value);
		if( littleEndian ) {
			a = Long.reverseBytes(a);
		}
		out.writeLong(a);
	}

	@Override
	public void writeFloat(float value) throws IOException {
		int a = Float.floatToIntBits(value);
		if( littleEndian ) {
			a = Integer.reverseBytes(a);
		}
		out.writeFloat(Float.intBitsToFloat(a));
	}

	@Override
	public void writeString(String value) throws IOException {

	}

	@Override
	public void writeS64(long value) throws IOException {

	}

	@Override
	public void writeS32(int value) throws IOException {

	}

	@Override
	public void writeU8(int value) throws IOException {

	}

	@Override
	public void writeArrayDouble(double[] storage, int size) throws IOException {

	}

	@Override
	public void writeArrayFloat(float[] storage, int size) throws IOException {

	}

	@Override
	public void writeArrayChar(char[] storage, int size) throws IOException {

	}

	@Override
	public void writeArrayByte(byte[] storage, int size) throws IOException {

	}
}
