package deepboof.io.torch7;

import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
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
		output.writeLong(a);
	}

	@Override
	public void writeFloat(float value) throws IOException {
		int a = Float.floatToIntBits(value);
		if( littleEndian ) {
			a = Integer.reverseBytes(a);
		}
		output.writeFloat(Float.intBitsToFloat(a));
	}

	@Override
	public void writeString(String value) throws IOException {
		writeS32(value.length());
		for (int i = 0; i < value.length(); i++) {
			byte b = (byte)value.charAt(i);
			writeU8(b);
		}
	}

	@Override
	public void writeS64(long value) throws IOException {
		if( littleEndian ) {
			output.writeLong( Long.reverseBytes(value) );
		} else {
			output.writeLong( value );
		}
	}

	@Override
	public void writeS32(int value) throws IOException {
		if( littleEndian ) {
			output.writeInt( Integer.reverseBytes(value) );
		} else {
			output.writeInt( value );
		}
	}

	@Override
	public void writeU8(int value) throws IOException {
		output.writeByte(value);
	}

	@Override
	public void writeArrayDouble(double[] storage, int size) throws IOException
	{
		ByteArrayOutputStream stream = new ByteArrayOutputStream(size*8);
		DataOutput encode = new DataOutputStream(stream);

		for (int i = 0; i < size; i++) {
			long a = Double.doubleToLongBits(storage[i]);

			if( littleEndian )
				encode.writeLong(Long.reverseBytes(a));
			else
				encode.writeLong(a);
		}

		output.write(stream.toByteArray());
	}

	@Override
	public void writeArrayFloat(float[] storage, int size) throws IOException
	{
		ByteArrayOutputStream stream = new ByteArrayOutputStream(size*4);
		DataOutput encode = new DataOutputStream(stream);

		for (int i = 0; i < size; i++) {
			int a = Float.floatToIntBits(storage[i]);

			if( littleEndian )
				encode.writeInt(Integer.reverseBytes(a));
			else
				encode.writeInt(a);
		}

		output.write(stream.toByteArray());
	}

	@Override
	public void writeArrayChar(char[] storage, int size) throws IOException
	{
		ByteArrayOutputStream stream = new ByteArrayOutputStream(size*2);
		DataOutput encode = new DataOutputStream(stream);

		for (int i = 0; i < size; i++) {
			if( littleEndian )
				encode.writeShort(Short.reverseBytes((short)storage[i]));
			else
				encode.writeShort(storage[i]);
		}

		output.write(stream.toByteArray());
	}

	@Override
	public void writeArrayByte(byte[] storage, int size) throws IOException
	{
		output.write(storage,0,size);
	}
}
