package deepboof.io.torch7;

import deepboof.io.torch7.struct.*;

import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Base class for serializing Torch data types.
 *
 * @author Peter Abeles
 */
public abstract class SerializeTorch7 {

	OutputStream stream;
	DataOutput output;

	protected boolean verbose = false;

	protected List<TorchObject> savedObjects = new ArrayList<>();

	public void serialize(TorchObject object , OutputStream stream ) throws IOException {
		List<TorchObject> objects = new ArrayList<>();
		objects.add( object );

		serialize(objects, stream);
	}

	public void serialize(List<TorchObject> objects , OutputStream stream ) throws IOException {
		this.stream = stream;
		this.output = new DataOutputStream(stream);

		this.savedObjects.clear();

		for (int i = 0; i < objects.size(); i++) {
			serializeObject(objects.get(i));
		}
	}

	protected void serializeObject(TorchObject object ) throws IOException {
		if( object instanceof TorchGeneric ) {
			TorchGeneric g = (TorchGeneric)object;
			if( g.torchName == null ) {
				writeType(TorchType.TABLE);
				serializeTable(g);
			} else {
				writeType(TorchType.TORCH);
				if( savedObjects.contains(object)) {
					writeS32(savedObjects.indexOf(object)+1);
				} else {
					writeS32(savedObjects.size() + 1);
					savedObjects.add(object);

					writeString("V " + g.version);
					writeString(g.torchName);
					if( g.map != null )
						serializeTable(g);
				}
			}
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
			serializeList((TorchList) object);
		} else {
			TorchReferenceable r = (TorchReferenceable)object;
			writeType(TorchType.TORCH);
			if( savedObjects.contains(object)) {
				writeS32(savedObjects.indexOf(object)+1);
			} else {
				writeS32(savedObjects.size()+1);
				savedObjects.add(object);

				writeString("V "+r.version);
				writeString(r.torchName);

				if( object instanceof TorchTensor ) {
					serializeTensor((TorchTensor)object);
					return;
				} else if( object instanceof TorchStorage ) {
					serializeStorage((TorchStorage)object);
					return;
				}
			}
			throw new RuntimeException("Support this type "+object.getClass().getSimpleName());
		}
	}

	protected void serializeStorage( TorchStorage storage ) throws IOException {

		switch( storage.torchName ) {
			case "torch.LongStorage"://{
				throw new IOException("LongStorage not yet supported");
//			}break;

			case "torch.FloatStorage":{
				writeS64(storage.size());
				writeArrayFloat(((TorchFloatStorage)storage).data,storage.size());
			}break;

			case "torch.DoubleStorage":{
				writeS64(storage.size());
				writeArrayDouble(((TorchDoubleStorage)storage).data,storage.size());
			}break;

			case "torch.ByteStorage":{
				writeS64(storage.size());
				writeArrayByte(((TorchByteStorage)storage).data,storage.size());
			}break;

			case "torch.CharStorage":{
				writeS64(storage.size()*2 + storage.size()%2);
				writeArrayChar(((TorchCharStorage)storage).data,storage.size());
			}break;

			default:
				throw new IOException("Unsupported storage type.  Please add support "+storage.torchName);
		}
	}

	protected void serializeTensor( TorchTensor object ) throws IOException {
		writeS32( object.shape.length );

		if( object.shape.length > 0 ) {
			// save the tensor's shape
			writeShape(object.shape);

			// compute and save the stride
			int stride[] = new int[ object.shape.length ];
			int N = 1;
			for (int i = object.shape.length-1; i >= 0; i--) {
				stride[i] = N;
				N *= object.shape[i];
			}
			writeShape(stride);

			// sub-tensor parameters
			writeS64(object.startIndex+1);
			serializeObject(object.storage);
		} else {
			// no idea what this is supposed to be
			writeS32(0);
			writeS64(0);
		}
	}

	protected void serializeTable(TorchGeneric object ) throws IOException {
		List<Object> keys = new ArrayList<>(object.map.keySet());

		writeS32(savedObjects.size()+1);
		savedObjects.add(object);
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
		writeS32(savedObjects.size()+1);
		savedObjects.add(object);
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
