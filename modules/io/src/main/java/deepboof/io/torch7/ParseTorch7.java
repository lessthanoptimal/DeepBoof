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

import deepboof.io.torch7.struct.*;

import java.io.*;
import java.util.*;

/**
 * <p>Parser for binary Torch 7 serialized objects.</p>
 *
 * Torch source code:<br>
 * <ul><li>torch7/File.lua</li></ul>
 *
 * @author Peter Abeles
 */
public abstract class ParseTorch7 {
	protected FileInputStream stream;
	protected DataInput input;

	protected Map<Integer,TorchReferenceable> masterTable = new HashMap<>();

	// verbose output?
	protected boolean verbose = false;

	public <T extends TorchObject>T parseOne(File file ) throws IOException {
		return (T)parse(file).get(0);
	}

	/**
	 * Parses the file, grabs the first element, and converts it into a Deep Boof object.
	 */
	public <T>T parseIntoBoof(File file ) throws IOException {
		return (T)ConvertTorchToBoofForward.convert(parseOne(file));
	}

	/**
	 * Parses serialized objects inside the specified file.  A list of the top level objects is returned
	 * @param file Input file
	 * @return List of {@link TorchObject}
	 * @throws IOException
	 */
	public List<TorchObject> parse(File file ) throws IOException {
		masterTable = new HashMap<>();
		stream = new FileInputStream(file);
		input = new DataInputStream(stream);

		List<TorchObject> list = new ArrayList<>();
		try {
			while (true) {
				list.add( parseNext(true) );
			}
		} catch( EOFException ignore ) {}

		stream.close();

		return list;
	}

	@SuppressWarnings("unchecked")
	private <T extends TorchObject> T parseNext(boolean useCached ) throws IOException {
		TorchType type = readType();
		if (verbose)
			System.out.println("========== Type = "+type);

		T found = null;
		switch( type ) {
			case TORCH: {
				int index = readS32();
				found = (T)lookupObject( index , useCached);
				if( found == null )
					found = (T)parseTorchObject(index);
			}break;

			case RECUR_FUNCTION: {
				int index = readS32();
				found = (T)lookupObject( index , useCached);
				if( found == null )
					found = (T)parseRecurFunction(index);
			}break;

			case TABLE: found =  (T)parseTable(); break;

			case STRING: found =  (T)parseString(); break;

			case BOOLEAN: found =  (T)parseBoolean(); break;

			case NUMBER: found =  (T)parseNumber(); break;

			case NIL: if (verbose){System.out.println("  ignoring nil");}break;

			default:
				if( verbose )
					System.out.println("Unsupported object type "+type);

		}

		if( found != null ) {
			if( found instanceof TorchReference ) {
				TorchReference r = (TorchReference)found;
				found = (T)masterTable.get(r.id);
			} else if( found instanceof TorchReferenceable ) {
				TorchReferenceable tr = (TorchReferenceable)found;
				masterTable.put(tr.index,tr);
			}
		}

		return found;
	}

	private TorchObject parseRecurFunction( int index ) throws IOException {
		String moo = readString();
		if( verbose ) {
			System.out.println("   not sure what to do with recur functions.  Here's their string:");
			System.out.println("   "+moo);
		}
		return parseNext(true);
	}

	private TorchObject parseTorchObject( int index ) throws IOException {

		int version = stringToVersionNumber(readString());
		String className = readString();

		if( verbose )
			System.out.println("  index = "+index+"  version = "+version+"  className = "+className);

		TorchReferenceable ret = null;

		if( className.startsWith("torch.")) {
			className = cudaToFloat(className);
			if (className.endsWith("Storage")) {
				ret = parseStorage(className);
			} else if( className.endsWith("Tensor")) {
				ret = parseTensor();
			}
		}

		if( ret == null ) {
			switch( className ) {

				default: {
					TorchGeneric t = new TorchGeneric();
					ret = t;

					TorchGeneric innerTable = parseNext(true);
					if (innerTable == null) {
						throw new RuntimeException("Probably an unsupported type.  Add support for " + className);
					}
					t.map = innerTable.map;
				}
			}
		}

		ret.index = index;
		ret.version = version;
		ret.torchName = className;

		return ret;
	}

	/**
	 * Looks up object in the master table and returns it if it already there.
	 * @param index Index in the table
	 * @param useCached If it has been configured to use cacheed objects
	 * @return The object or null if it was not found
	 */
	private TorchObject lookupObject( int index , boolean useCached ) {
		if( useCached && masterTable.containsKey(index) ) {
			if( verbose )
				System.out.println("reference index = "+index);
			TorchReference ret = new TorchReference();
			ret.id = index;
			return ret;
		}
		return null;
	}

	private String cudaToFloat(String className) {
		if( className.equals("torch.CudaStorage") ) {
			className = "torch.FloatStorage";
		} else if( className.equals("torch.CudaTensor") ) {
			className = "torch.FloatTensor";
		}
		return className;
	}

	private TorchTensor parseTensor() throws IOException {
		TorchTensor t = new TorchTensor();
		int dimension = readS32();
		if (dimension != 0) {
			// read the dimension
			t.shape = readShape(dimension);
			if( verbose )
				System.out.println("   shape dimension = "+t.shape.length);

			// read the stride.  No need to save since it can be computed from the shape
			readShape(dimension);
			t.startIndex = (int)readS64()-1; // Lua is 1 index not 0
			t.storage = parseNext(true);
			if( verbose && (t.storage.size() != t.length() || t.startIndex != 0) ) {
				System.out.println("subtensor.  storage "+t.storage.size()+"  tensor "+t.length()+"  offset "+t.startIndex);
			}
		} else {
			int a = readS32();
			long b = readS64();

			if( verbose )
				System.out.println("    no dimension.  Weird variable "+a+" "+b);
		}
		return t;
	}

	private TorchStorage parseStorage( String name ) throws IOException {
		int size = (int)readS64();

		TorchStorage out;
		switch( name ) {
			case "torch.LongStorage":{
				TorchLongStorage t = new TorchLongStorage(size);
				readArrayLong(size,t.data);
				out = t;
			}break;
			case "torch.FloatStorage":{
				TorchFloatStorage t = new TorchFloatStorage(size);
				readArrayFloat(size,t.data);
				out = t;
			}break;

			case "torch.DoubleStorage":{
				TorchDoubleStorage t = new TorchDoubleStorage(size);
				readArrayDouble(size,t.data);
				out = t;
			}break;

			case "torch.ByteStorage":{
				TorchByteStorage t = new TorchByteStorage(size);
				readArrayByte(size,t.data);
				out = t;
			}break;

			case "torch.CharStorage":{
				TorchCharStorage t = new TorchCharStorage(size/2+size%2);
				readArrayChar(size,t.data);
				out = t;
			}break;

			default:
				throw new IOException("Unsupported storage type.  Please add support "+name);
		}

		if( verbose ) {
			out.printSummary();
		}
		return out;
	}

	private void printNextHex( int N ) throws IOException {
		for (int i = 0; i < N; i++) {
			System.out.printf("%02x ",(int)(input.readByte()&0xFF));
		}
		System.out.println();
	}

	private TorchObject parseTable() throws IOException {
		int index = readS32();
		int size = readS32();

		if( verbose ) {
			System.out.println("  idx = " + index);
			System.out.println("  size = " + size);
		}

		Map<Object,TorchObject> map = new HashMap<>();

		for (int i = 0; i < size; i++) {
			Object key;
			TorchObject o_key = parseNext(true);
			if( o_key instanceof TorchString ) {
				key = ((TorchString)o_key).message;
			} else if( o_key instanceof TorchNumber ) {
				key = ((TorchNumber)o_key).value;
			} else {
				throw new RuntimeException("Add support for "+o_key);
			}

			TorchObject value = parseNext(true);

			// adjust the type from cuda to float
			if( key.equals("_type") ) {
				TorchString s = (TorchString)value;
				s.message = cudaToFloat( s.message );
			}

			if( map.put(key,value) != null )
				throw new RuntimeException("Probably a bug in the parser.  Same key assigned twice");
		}

		if( size > 0 && isList(map) ) {
			List<Double> listKeys = new ArrayList<>();
			listKeys.addAll( (Collection)map.keySet() );
			Collections.sort(listKeys);

			TorchList t = new TorchList();
			t.index = index;
			for (int i = 0; i < listKeys.size(); i++ ) {
				Double number = listKeys.get(i);
				int value = number.intValue();
				if( value != i+1 )
					throw new RuntimeException("Not actually a complete sequential list");
				t.list.add( map.get(number));
			}
			return t;
		} else {
			TorchGeneric t = new TorchGeneric();
			t.map = map;
			t.index = index;
			return t;
		}
	}

	private boolean isList( Map<Object,TorchObject> map ) {
		for( Object o : map.keySet() ) {
			if( !(o instanceof Double) )
				return false;
		}
		return true;
	}

	private TorchObject parseString() throws IOException {
		TorchString ret = new TorchString();
		ret.message = readString();
		if( verbose )
			System.out.println("   "+ret.message);
		return ret;
	}

	private TorchObject parseBoolean() throws IOException {
		TorchBoolean ret = new TorchBoolean();
		ret.value = readBoolean();
		if( verbose )
			System.out.println("   "+ret.value);
		return ret;
	}

	private TorchObject parseNumber() throws IOException {
		TorchNumber ret = new TorchNumber();
		ret.value = readDouble();
		return ret;
	}

	private void parseRecurFunction() throws IOException {
		int index = readS32();
		throw new IOException("Not supported yet.  RecurFunction");
	}

	public int stringToVersionNumber( String line ) {
		if( line.length() < 3 || line.charAt(0) != 'V' || line.charAt(1) != ' ')
//			return 0;
			throw new RuntimeException("Old format.  Add support for this");

		String substring = line.substring(2,line.length());

		return Integer.parseInt(substring);
	}

	public abstract int[] readShape( int dimension ) throws IOException;

	public abstract TorchType readType() throws IOException;

	public abstract boolean readBoolean() throws IOException ;

	public abstract double readDouble() throws IOException;

	public abstract float readFloat() throws IOException;

	public abstract String readString() throws IOException;

	public abstract long readS64() throws IOException;

	public abstract int readS32() throws IOException;

	public abstract int readU8() throws IOException;

	public abstract void readArrayDouble( int size , double[] storage ) throws IOException;

	public abstract void readArrayFloat( int size , float[] storage ) throws IOException;

	public abstract void readArrayChar( int size , char[] storage ) throws IOException;

	public abstract void readArrayByte( int size , byte[] storage ) throws IOException;

	public abstract void readArrayLong( int size , long[] storage ) throws IOException;

	public boolean isVerbose() {
		return verbose;
	}

	public ParseTorch7 setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}

}
