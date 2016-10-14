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

package deepboof.graph;

import deepboof.Function;
import deepboof.Tensor;
import deepboof.misc.TensorFactory;
import deepboof.misc.TensorOps;
import org.ddogleg.struct.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static deepboof.misc.TensorOps.WI;

/**
 * Processes a sequence of forward functions.  Any non-cyclical graph with a single
 * input and a single output can be processed by this function.  The list of functions passed in to the constructor
 * is assumed to have already been ordered.
 *
 * @author Peter Abeles
 */
public class FunctionSequence<T extends Tensor<T>, F extends Function<T>>
{
	// Sequence of functions which have been ordered so that the pre-requisites are meet by nodes previously
	// in the list.
	protected List<Node<T,F>> sequence = new ArrayList<>();
	// Map to provide quick and easy lookup of
	protected Map<String,Node<T,F>> lookup = new HashMap<>();

	// map linking output storage for each node by name.  data0 = function output, data1 = merge output
	protected Map<String,Tuple2<T,T>> outputStorage = new HashMap<>();

	// used to create tensors
	protected TensorFactory<T> factory;

	boolean verbose = false;

	/**
	 * Configures the sequence
	 *
	 * @param sequence Sequence of functions which has been ordered to meet pre-requisites.
	 * @param type Type of tensor
	 */
	public FunctionSequence(List<Node<T,F>> sequence, Class<T> type ) {
		this.sequence = sequence;

		for( Node<T,F> n : sequence ) {
			if( lookup.containsKey(n.name ))
				throw new IllegalArgumentException("Conflict.  Multiple nodes with the same name.  "+n.name);
			lookup.put(n.name,n);
		}

		factory = new TensorFactory<>(type);
	}

	/**
	 * Initialize and declare memory for all nodes in the network given the shape of the input tensor
	 *
	 * @param inputShape Shape of input tensor.
	 */
	public void initialize(int[] inputShape ) {
		initializeSequence(inputShape);
	}

	/**
	 * Run through the sequence initializing each node using the shape of the output of its inputs
	 *
	 * @param inputShape Shape of input tensor.
	 */
	private void initializeSequence(int[] inputShape) {
		if( sequence.get(0).sources.size() != 0 )
			throw new RuntimeException("Input sequence can't have a source address!");

		List<int[]> inputs = new ArrayList<>();
		sequence.get(0).function.initialize(inputShape);
		outputStorage.put( sequence.get(0).name, new Tuple2<>(factory.create(),factory.create()) );
		if( verbose ) {
			System.out.println("ROOT ========= " + sequence.get(0).name);
			printOutput(sequence.get(0), inputShape);
		}

		for (int i = 1; i < sequence.size(); i++) {
			Node<T,F> node = sequence.get(i);
			if( verbose )
				System.out.println("============== "+node.name);
			outputStorage.put( node.name, new Tuple2<>(factory.create(),factory.create()) );
			if( node.sources.size() == 0 )
				throw new RuntimeException("No sources!  Node = "+node.name);

			// collect the size of all the inputs for this node
			inputs.clear();
			for (int j = 0; j < node.sources.size(); j++) {
				InputAddress addr = node.sources.get(j);

				Node<T,F> src = lookup.get(addr.nodeName);
				if( src == null )
					throw new RuntimeException("Can't find input node from name.  Bad network");
				inputs.add( src.function.getOutputShape() );
				if( verbose )
					System.out.println("   input addr "+addr.nodeName);
			}

			// If just one input then it goes to a function, otherwise it gets combined and then passed to the function
			if( inputs.size() == 1 ) {
				node.function.initialize(inputs.get(0));
				if( verbose )
					printOutput(node,inputs.get(0));
			} else {
				if( node.combine == null )
					throw new RuntimeException("Must specify a combine operator if there are multiple sources");
				node.combine.initialize(inputs);
				node.function.initialize(node.combine.getOutputShape());
				if( verbose )
					printOutput(node,node.combine.getOutputShape());
			}
		}
	}

	private void printOutput( Node<T,F> node , int[] input  ) {
		int[] output = node.function.getOutputShape();
		String sin = TensorOps.toStringShape(input);
		String sout = TensorOps.toStringShape(output);
		System.out.printf("%30s input %25s  out = %25s\n",node.function.getClass().getSimpleName(),sin,sout);
	}


	/**
	 * Declare and save output tensors for each node and combine function
	 */
	private void declareOutputStorage( int numBatch ) {
		// input and output is provided if size of one and it's impossible for it to have a combine function
		if( sequence.size() == 1 ) {
			return;
		}

		// Declare storage for output from each node.   The last node doesn't need additional storage
		for (int i = 0; i < sequence.size(); i++) {
			Node<T,F> node = sequence.get(i);

			Tuple2<T,T> storage = outputStorage.get(node.name);
			if( i==0 || node.sources.size() == 1 ) {
				if( i != sequence.size()-1 )
					storage.data0.reshape(WI(numBatch,node.function.getOutputShape()));
				storage.data1 = null;
			} else {
				// don't declare memory for output for the last node since it will be provided
				if( i != sequence.size()-1 )
					storage.data0.reshape(WI(node.function.getOutputShape()));
				// however, the last node could still need storage for combining inputs
				storage.data1.reshape(WI(node.combine.getOutputShape()));
			}
		}
	}

	/**
	 * Specify the parameters for each node in the network
	 * @param nodeParameters Map where the key is the function/node name and the value is the parameters for that node
	 */
	public void setParameters( Map<String,List<T>> nodeParameters ) {
		for (int i = 0; i < sequence.size(); i++){
			Node<T, F> node = sequence.get(i);

			List<T> parameters = nodeParameters.get(node.name);
			if( parameters != null ) {
				node.function.setParameters(parameters);
			}
		}
	}

	/**
	 * Process the input tensor and compute the output tensor by feeding the results through the network.  Must
	 * call {@link #initialize} once with the same shape as the input tensor.  Must also call {@link #setParameters}
	 * @param input Input tensor
	 * @param output Storage for output tensor.
	 */
	public void process( T input , T output ) {
		if( sequence.size() == 1 ) {
			sequence.get(0).function.forward(input, output);
			return;
		}
		// Adjust the size of inner tensors
		declareOutputStorage(input.length(0));

		// TODO more meaningful error messages that say which node in the sequence it crashed on
		// Handle the head node.  No input node
		{
			Node<T,F> node = sequence.get(0);
			Tuple2<T,T> storage = outputStorage.get(node.name);
			node.function.forward(input, storage.data0 );
		}

		// Handle all the inner nodes in the sequence.
		List<T> inputs = new ArrayList<>();
		for (int i = 1; i < sequence.size() - 1; i++) {
			Node<T,F> node = sequence.get(i);
			Tuple2<T,T> nodeOutput = outputStorage.get(node.name);

			// Collect input tensors from parent nodes
			inputs.clear();
			for (int j = 0; j < node.sources.size(); j++) {
				InputAddress addr = node.sources.get(j);
				inputs.add( outputStorage.get(addr.nodeName).data0 );
			}

			// Process the inputs now and store in output
			if( node.sources.size() == 1 ) {
				node.function.forward(inputs.get(0),nodeOutput.data0);
			} else {
				node.combine.combine(inputs,nodeOutput.data1);
				node.function.forward(nodeOutput.data1,nodeOutput.data0);
			}
		}

		// Handle the tail node.  No output node
		{
			Node<T, F> node = sequence.get(sequence.size() - 1);
			inputs.clear();
			for (int j = 0; j < node.sources.size(); j++) {
				InputAddress addr = node.sources.get(j);
				inputs.add(outputStorage.get(addr.nodeName).data0);
			}

			if( node.sources.size() == 1 ) {
				node.function.forward(inputs.get(0),output);
			} else {
				Tuple2<T,T> nodeOutput = outputStorage.get(node.name);
				node.combine.combine(inputs,nodeOutput.data1);
				node.function.forward(nodeOutput.data1,output);
			}
		}
	}

	public List<Node<T, F>> getSequence() {
		return sequence;
	}

	public T getNodeOutput(int index ) {
		return outputStorage.get( sequence.get(index).name ).data0;
	}

	public int[] getOutputShape() {
		return sequence.get( sequence.size()-1 ).function.getOutputShape();
	}

	public Class<T> getTensorType() {
		return factory.getTensorType();
	}
}
