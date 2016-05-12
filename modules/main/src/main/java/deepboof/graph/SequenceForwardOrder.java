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

import java.util.*;

/**
 * Orders an unsorted list of nodes so that they can be processed in sequence and have all of their dependencies meet
 * prior to being invoked.  The graph must not have islands or cycles.  Some sanity checking is done to
 * ensure that these preconditions are meet, but not all situations are currently caught.
 * 
 * <pre>
 * Assumptions:
 * - One input node
 * - One output node
 * - No islands
 * - No cycles
 * </pre>
 * @author Peter Abeles
 */
public class SequenceForwardOrder {

	// Input sequence augmented with additional data.  Assumed to be unordered
	List<NodeData> sequence = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param list Input list, not modified.
	 */
	public SequenceForwardOrder(List<Node<?,?>> list ) {
		// used to quickly look up nodes by name
		Map<String,NodeData> lookup = new HashMap<>();

		// create a list of nodes with auxiliary data
		for( Node<?,?> n : list ) {
			NodeData d = new NodeData(n);
			sequence.add(d);
			lookup.put( n.name , d );
		}

		// fill in next and previous lists in each node
		for (int i = 0; i < sequence.size(); i++) {
			NodeData n = sequence.get(i);

			for (int j = 0; j < n.node.sources.size(); j++) {
				NodeData c = lookup.get( n.node.sources.get(j).nodeName );
				n.previous.add( c );
				c.next.add(n);
			}
		}
	}


	/**
	 * Orders list to ensure sequential forward ordering of nodes.
	 *
	 * @return Ordered list of nodes.
	 */
	public List<Node<?,?>> putIntoForwardOrder() {
		assignDepth();

		// See if any nodes were not traversed.  If that's the code then there are disconnected nodes
		for( int i = 0; i < sequence.size(); i++ ) {
			NodeData n = sequence.get(i);

			if( n.depth == Integer.MAX_VALUE ) {
				throw new RuntimeException("Disconnected node from graph "+n.node.name);
			}
		}

		// use a copy so that the input list isn't modified
		List<NodeData> copy = new ArrayList<>(sequence);

		// Sort the list based on depth-+
		Collections.sort(copy,new CompareWithDepth());

		List<Node<?,?>> ordered = new ArrayList<>();
		for (int i = 0; i < copy.size(); i++) {
			ordered.add( copy.get(i).node );
		}

		return ordered;
	}

	/**
	 * Assigns a depth from the input node for all the elements in the graph.  Depth is defined as the distance
	 * of the longest path to the node.
	 */
	protected void assignDepth() {
		resetNodeInfo();

		NodeData input = findInput();
		input.depth = 0;
		List<NodeData> layer = new ArrayList<>();
		List<NodeData> nextLayer = new ArrayList<>();

		layer.addAll(input.next);

		for (int i = 0; i < input.next.size(); i++) {
			NodeData c = input.next.get(i);
			if( c.depth == Integer.MAX_VALUE ) {
				c.depth = 1;
			} else {
				throw new RuntimeException("Input node connects to a child node more than once! "+c.node.name);
			}
		}

		int depth = 1;
		while( !layer.isEmpty() ) {
			nextLayer.clear();
			for (int i = 0; i < layer.size(); i++) {
				NodeData n = layer.get(i);

				// Set the depth of all of its children
				for (int j = 0; j < n.next.size(); j++) {
					NodeData c = n.next.get(j);
					if( c.depth == Integer.MAX_VALUE ) {
						// have all of it's parents been assigned a depth?  If not wait.  This will ensure that
						// it's depth is the depth of the longest path
						boolean allAssigned = true;
						for (int k = 0; k < c.previous.size(); k++) {
							if( c.previous.get(k).depth == Integer.MAX_VALUE ) {
								allAssigned = false;
								break;
							}
						}

						if( allAssigned ) {
							c.depth = depth + 1;
							nextLayer.add(c);
						}
					}
				}

				// Sanity check the graph
				for (int j = 0; j < n.previous.size(); j++) {
					NodeData c = n.previous.get(j);
					if( c.depth == Integer.MAX_VALUE ) {
						throw new RuntimeException("An input to this node has not been traversed.  Cycle or other graph error. "+c.node.name);
					}
				}
			}

			List<NodeData> tmp = layer;
			layer = nextLayer;
			nextLayer = tmp;

			depth++;
		}
	}

	/**
	 * Finds the input node.  Throws an error if there isn't one and only one input node
	 * @return Input node
	 */
	protected NodeData findInput() {
		NodeData found = null;
		for (int i = 0; i < sequence.size(); i++) {
			NodeData n = sequence.get(i);
			if( n.node.sources.isEmpty() ) {
				if( found != null )
					throw new RuntimeException("Found multiple input nodes");
				found = n;
			}
		}
		if( found == null )
			throw new RuntimeException("No input node found");
		return found;
	}

	private void resetNodeInfo() {
		for( int i = 0; i < sequence.size(); i++ ) {
			sequence.get(i).reset();
		}
	}

	private static class CompareWithDepth implements Comparator<NodeData> {

		@Override
		public int compare(NodeData a, NodeData b ) {
			if( a.depth < b.depth ) {
				return -1;
			} else if( a.depth > b.depth ) {
				return 1;
			} else {
				// use the name to break a tie
				return a.node.name.compareTo(b.node.name);
			}
		}
	}

	public static class NodeData
	{
		Node<?,?> node;
		// list of nodes which this node provides the input to
		List<NodeData> next = new ArrayList<>();
		// list of nodes which this node uses as input
		List<NodeData> previous = new ArrayList<>();

		// the minimum distance from the input node
		int depth;

		public NodeData(Node<?, ?> node) {
			this.node = node;
			reset();
		}

		public void reset() {
			depth = Integer.MAX_VALUE;
		}
	}
}
