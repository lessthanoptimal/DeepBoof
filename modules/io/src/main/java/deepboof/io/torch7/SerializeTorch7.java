package deepboof.io.torch7;

import deepboof.io.torch7.struct.*;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public abstract class SerializeTorch7 {

	OutputStream stream;

	protected boolean verbose = false;

	protected int counterObject;

	public void serialize(List<TorchObject> objects , OutputStream stream ) throws IOException {
		this.stream = stream;

		this.counterObject = 0;

		for (int i = 0; i < objects.size(); i++) {
			serializeObject(objects.get(i));
		}
	}

	protected void serializeObject(TorchObject object ) throws IOException {
		if( object instanceof TorchGeneric ) {
			writeType(TorchType.TABLE);
			serializeGeneric((TorchGeneric)object);
		} else if( object instanceof TorchString ) {
			writeType(TorchType.STRING);
			writeString(((TorchString)object).message);
		} else if( object instanceof TorchNumber ) {
			writeType(TorchType.NUMBER);
			writeDouble(((TorchNumber)object).value);
		} else if( object instanceof TorchBoolean) {
			writeType(TorchType.BOOLEAN);
			writeBoolean(((TorchBoolean)object).value);
		} else if( object instanceof TorchList) {
			writeType(TorchType.TABLE);
			serializeList((TorchList)object);
		} else {
			writeType(TorchType.TORCH);
			// TODO WRITE
		}
	}

	protected void serializeGeneric( TorchGeneric object ) throws IOException {
		List<Object> keys = new ArrayList<>(object.map.keySet());

		writeS32(counterObject++);
		writeS32(keys.size());

		for (Object key : keys) {

			if (key instanceof String) {
				TorchString t = new TorchString();
				t.message = (String) key;
				serializeObject(t);
			} else if (key instanceof Double) {
				TorchNumber t = new TorchNumber();
				t.value = (double) key;
				serializeObject(t);
			} else {
				throw new RuntimeException("Keys of this type not yet supported or this is a bug. " + key.getClass().getSimpleName());
			}

			serializeObject(object.map.get(key));
		}
	}

	protected void serializeList( TorchList object ) throws IOException {
		writeS32(counterObject++);
		writeS32(object.list.size());

		for ( int i = 0; i < object.list.size(); i++ ) {

			TorchNumber key = new TorchNumber();
			key.value = i+1;
			serializeObject(key);
			serializeObject(object.list.get(i));
		}
	}

	public abstract void writeShape( int[] shape ) throws IOException;

	public abstract void writeType( TorchType type ) throws IOException;

	public abstract void writeBoolean( boolean value ) throws IOException ;

	public abstract void writeDouble( double value ) throws IOException;

	public abstract void writeFloat( float value ) throws IOException;

	public abstract void writeString( String value ) throws IOException;

	public abstract void writeS64( long value ) throws IOException;

	public abstract void writeS32( int value ) throws IOException;

	public abstract void writeU8( int value ) throws IOException;

	public abstract void writeArrayDouble( double[] storage , int size ) throws IOException;

	public abstract void writeArrayFloat( float[] storage , int size ) throws IOException;

	public abstract void writeArrayChar( char[] storage , int size ) throws IOException;

	public abstract void writeArrayByte(  byte[] storage , int size ) throws IOException;

	public boolean isVerbose() {
		return verbose;
	}

	public SerializeTorch7 setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}
}
