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

package deepboof.misc;

import deepboof.Tensor;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_F64;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class TensorOps {

	/**
	 * Convenience function for wrapping passed in elements into a list
	 */
	public static <T> List<T> WT(T ...elements ) {
		List<T> list = new ArrayList<>();

		for (int i = 0; i < elements.length; i++) {
			list.add(elements[i]);
		}

		return list;
	}

	/**
	 * Convenience function for making it easy to create an array of ints
	 */
	public static int[] WI( int ... elements ) {
		return elements;
	}

	/**
	 * Convenience function for making it easy to create an array of ints
	 *
	 * @return [ a , elements[0] , ..., elements[N-1] ]
	 */
	public static int[] WI( int a , int[] elements ) {
		int out[] = new int[1+elements.length];
		out[0] = a;
		System.arraycopy(elements,0,out,1,elements.length);

		return out;
	}

	/**
	 * Convenience function for making it easy to create an array of ints
	 *
	 * @return [ elements[0] , ..., elements[N-1], a ]
	 */
	public static int[] WI( int[] elements , int a ) {
		int out[] = new int[1+elements.length];
		System.arraycopy(elements,0,out,0,elements.length);
		out[elements.length] = a;

		return out;
	}

	/**
	 * Truncates the head (element 0) from the array
	 */
	public static int[] TH(int[] elements ) {
		int[] out = new int[ elements.length-1 ];
		System.arraycopy(elements,1,out,0,out.length);
		return out;
	}

	public static List<int[]> WI( int a , List<int[]> list ) {
		List<int[]> output = new ArrayList<int[]>();

		for( int[] elements : list ) {
			output.add( WI(a,elements));
		}

		return output;
	}

	/**
	 * Adds a dimension to the input tensor.  Returns a new tensor which references
	 * the same data as the original.
	 *
	 * Input Shape = [5 4 3]
	 * Output Shape = [1 5 4 3]
	 */
	public static <T extends Tensor>T AD( T input ) {
		T out;
		if( input instanceof Tensor_F64 ) {
			out = (T)new Tensor_F64();
		} else {
			throw new RuntimeException("Unsupported type");
		}
		out.shape = WI(1,input.shape);
		out.setData(input.getData());
		out.computeStrides();
		return out;

	}

	/**
	 * Returns the total length of all the tensors in the list summed together
	 *
	 * @param shapes List of tensor shapes
	 * @return Sum of tensor lengths
	 */
	public static int sumTensorLength( List<int[]> shapes ) {
		int total = 0;

		for( int i =0; i < shapes.size(); i++ ) {
			total += tensorLength(shapes.get(i));
		}

		return total;
	}

	/**
	 * Returns the total length of one tensor
	 *
	 * @param shape shape of a tensor
	 * @return length of the Tensor's data arrayn
	 */
	public static int tensorLength( int... shape ) {
		if( shape.length == 0 )
			return 0;
		int N = 1;
		for (int i = 0; i < shape.length; i++) {
			N *= shape[i];
		}
		return N;
	}

	/**
	 * Compares a list of tensors shape's against each other.
	 * This simply invokes {@link #checkShape(String, int, int[], int[], boolean)}
	 *
	 * @param which String describing which variables are being checked
	 * @param expected List of expected tensors
	 * @param actual List of actual tensors.  Axis 0 is optionally ignored here, see ignoreAxis0
	 * @param ignoreAxis0 true to ignore axis 0
	 */
	public static void checkShape(String which, List<int[]> expected , List<Tensor<?>> actual , boolean ignoreAxis0 )
	{
		if( expected.size() != actual.size() )
			throw new IllegalArgumentException(
					which+": Unexpected number of tensors. "+expected.size()+" vs "+actual.size());

		for (int i = 0; i < expected.size(); i++) {
			int[] e = expected.get(i);
			int[] a = actual.get(i).getShape();

			checkShape(which, i, e, a, ignoreAxis0);
		}
	}

	/**
	 * Checks to see if the two tensors have the same shape
	 * @param a tensor
	 * @param b tensor
	 */
	public static void checkShape( Tensor_F64 a , Tensor_F64 b ) {
		if( a.shape.length != b.shape.length ) {
			throw new IllegalArgumentException("Dimension of tensors do not match. "+a.shape.length+" "+b.shape.length);
		}
		for (int i = 0; i < a.shape.length; i++) {
			int da = a.shape[i];
			int db = b.shape[i];

			if( da != db ) {
				throw new IllegalArgumentException("dimension "+i+"  does not match.  "+da+"  "+db);
			}
		}
	}

	/**
	 * Checks to see if the two tensors have the same shape
	 * @param a tensor
	 * @param b tensor
	 */
	public static void checkShape( Tensor_F32 a , Tensor_F32 b ) {
		if( a.shape.length != b.shape.length ) {
			throw new IllegalArgumentException("Dimension of tensors do not match. "+a.shape.length+" "+b.shape.length);
		}
		for (int i = 0; i < a.shape.length; i++) {
			int da = a.shape[i];
			int db = b.shape[i];

			if( da != db ) {
				throw new IllegalArgumentException("dimension "+i+"  does not match.  "+da+"  "+db);
			}
		}
	}

	/**
	 * Checks to see if the two tensors have the same shape, with the option to ignore the first axis for the 'actual'
	 * shape.   The first axis is typically the mini-batch, but the expected value might not include the mini-batch
	 * since the size of the mini-batch is determined later on.
     * Throws an {@link IllegalArgumentException} if they don't match.
	 *
	 * @param which String describing which variable is being checked
	 * @param tensor Index of the tensor in a tensor list.  Used to provide a more detailed error message.
	 *               If &lt; 0 then this is ignored
	 * @param expected Expected shape.
	 * @param actual Actual shape.  Axis 0 is optionally ignored.
	 * @param ignoreAxis0 If true it will ignore the first dimension in expected
	 */
	public static void checkShape(String which, int tensor, int[] expected, int[] actual, boolean ignoreAxis0 ) {

		if( ignoreAxis0 ) {
			if (expected.length + 1 != actual.length) {
				String header = tensor >= 0 ? which + ":  Tensor[" + tensor + "] " : which + ": ";
				throw new IllegalArgumentException(header + " dimension doesn't match, expected = "
						+ (expected.length + 1) + " found = " + actual.length);
			} else {
				for (int i = 0; i < expected.length; i++) {
					if (expected[i] != actual[i+1]) {
						String header = tensor >= 0 ? which + ":  Tensor[" + tensor + "] " : which + ": ";

						throw new IllegalArgumentException(header + " shapes don't match, expected = "
								+ toStringShape(expected) + ", found = " + toStringShapeA(actual));
					}
				}
			}
		} else {
			if (expected.length != actual.length) {
				String header = tensor >= 0 ? which + ":  Tensor[" + tensor + "] " : which + ": ";
				throw new IllegalArgumentException(header + " dimension doesn't match, expected = "
						+ expected.length + " found = " + actual.length);
			} else {
				for (int i = 0; i < expected.length; i++) {
					if (expected[i] != actual[i]) {
						String header = tensor >= 0 ? which + ":  Tensor[" + tensor + "] " : which + ": ";

						throw new IllegalArgumentException(header + " shapes don't match, expected = "
								+ toStringShape(expected) + ", found = " + toStringShape(actual));
					}
				}
			}
		}
	}

	public static String toStringShapeA( int []shape ) {
		String out = "( * , ";
		for (int i = 1; i < shape.length; i++) {
			out += shape[i] +" , ";
		}
		return out + ")";
	}

	public static String toStringShape( int []shape ) {
		String out = "( ";
		for (int i = 0; i < shape.length; i++) {
			out += shape[i] +" , ";
		}
		return out + ")";
	}

	/**
	 * <p>Computes the number of elements for an inner portion of the tensor starting at
	 * the specified index and going outside</p>
	 *
	 * Example:<br>
	 * Tensor shape = (d[0], ... , d[K-1]).  Then if start dimen is 2, the output will
	 * be the product of d[2] to d[K-1].
	 */
	public static int outerLength(int[] shape , int startDimen ) {
		if( startDimen >= shape.length )
			return 0;

		int D = 1;
		for (int i = startDimen; i < shape.length; i++) {
			D *= shape[i];
		}
		return D;
	}

	public static File pathToRoot() {
		File active = new File(".").getAbsoluteFile();

		while( active != null ) {
			boolean foundModules = false;
			boolean foundExamples = false;
			boolean foundSettings = false;

			File[] children = active.listFiles();
			if( children == null )
				break;

			for( File d : children ) {
				if( d.isDirectory() && d.getName().endsWith("modules")) {
					foundModules = true;
				}
				if( d.isDirectory() && d.getName().endsWith("examples")) {
					foundExamples = true;
				}
				if( d.isFile() && d.getName().equals("settings.gradle")) {
					foundSettings = true;
				}
			}

			if( foundModules && foundExamples && foundSettings ) {
				return active;
			} else {
				active = active.getParentFile();
			}
		}
		throw new RuntimeException("Cant find the project root directory");

	}

	/**
	 * Computes the sum of all the elements in the tensor
	 * @param tensor Tensor
	 */
	public static double elementSum( Tensor tensor ) {
		if( tensor instanceof Tensor_F64 ) {
			return TensorOps_F64.elementSum( (Tensor_F64)tensor );
		} else if( tensor instanceof Tensor_F32 ) {
			return TensorOps_F32.elementSum( (Tensor_F32)tensor );
		} else {
			throw new IllegalArgumentException("Support not added yet for this tensor type");
		}
	}

	public static void fill( Tensor t , double value ) {
		if( t instanceof Tensor_F64 ) {
			TensorOps_F64.fill( (Tensor_F64)t, value );
		} else if( t instanceof Tensor_F32 ) {
			TensorOps_F32.fill( (Tensor_F32)t, (float)value );
		} else {
			throw new IllegalArgumentException("Support not added yet for this tensor type");
		}
	}

	public static void boundSpatial( int bounds[] , int rows , int cols ) {
		if( bounds[0] < 0 ) bounds[0] = 0;
		if( bounds[1] < 0 ) bounds[1] = 0;
		if( bounds[2] > rows ) bounds[2] = rows;
		if( bounds[3] > cols ) bounds[3] = cols;
	}
}
