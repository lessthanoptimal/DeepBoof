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

import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Peter Abeles
 */
@SuppressWarnings("unchecked")
public class TestSequenceForwardOrder {

	Random rand = new Random(234);

	/**
	 * Makes sure the constructor correctly set up the network
	 */
	@Test
	public void constructor_one() {
		List<Node> list = new ArrayList<>();

		list.add( create("first") );

		SequenceForwardOrder alg = new SequenceForwardOrder((List)list);

		assertEquals(alg.sequence.size(),1);

		SequenceForwardOrder.NodeData n = alg.sequence.get(0);
		assertEquals(0,n.next.size());
		assertEquals(0,n.previous.size());
	}

	@Test
	public void constructor_many() {
		List<Node> list = new ArrayList<>();

		list.add( create("1") );
		list.add( create("2") );
		list.add( create("3") );

		list.get(1).sources.add( new InputAddress("1"));
		list.get(2).sources.add( new InputAddress("2"));

		SequenceForwardOrder alg = new SequenceForwardOrder((List)list);

		assertEquals(alg.sequence.size(),3);

		check(alg.sequence.get(0),"1",1,0);
		check(alg.sequence.get(1),"2",1,1);
		check(alg.sequence.get(2),"3",0,1);

		connected(alg.sequence.get(0),alg.sequence.get(1));
		connected(alg.sequence.get(1),alg.sequence.get(2));
	}

	@Test
	public void putIntoForwardOrder_line() {
		for (int length = 1; length < 5; length++) {
			List<Node> ordered = createLineSequence(length);
			List<Node> shuffled = new ArrayList<>(ordered);
			Collections.shuffle(shuffled, rand);

			SequenceForwardOrder alg = new SequenceForwardOrder((List) shuffled);

			List<Node<?,?>> found = alg.putIntoForwardOrder();

			for (int i = 0; i < length; i++) {
				assertTrue(ordered.get(i)==found.get(i));
			}
		}
	}

	@Test
	public void putIntoForwardOrder_branch() {
		List<Node> ordered = createBranchA();

		// change the order a bunch to test more variants
		for (int trial = 0; trial < 10; trial++) {
			List<Node> shuffled = new ArrayList<>(ordered);
			Collections.shuffle(shuffled, rand);

			SequenceForwardOrder alg = new SequenceForwardOrder((List) shuffled);

			List<Node<?,?>> found = alg.putIntoForwardOrder();

			for (int i = 0; i < ordered.size(); i++) {
				assertTrue(ordered.get(i)==found.get(i));
			}
		}
	}

	/**
	 * Tests the ability to assign depth for a simple line graph
	 */
	@Test
	public void assignDepth_line() {
		for (int length = 1; length < 4; length++) {
			List<Node> ordered = createLineSequence(length);
			List<Node> shuffled = new ArrayList<>(ordered);
			Collections.shuffle(shuffled,rand);

			SequenceForwardOrder alg = new SequenceForwardOrder((List)shuffled);

			alg.assignDepth();

			for (int j = 0; j < length; j++) {
				SequenceForwardOrder.NodeData d = alg.sequence.get(j);
				int expected = ordered.indexOf(d.node);
				assertEquals(expected,d.depth);
			}
		}
	}

	private List<Node> createLineSequence(int length) {
		List<Node> ordered = new ArrayList<>();

		for (int j = 0; j < length; j++) {
			ordered.add( create(""+j) );
		}
		for (int j = 1; j < length; j++) {
			ordered.get(j).sources.add( new InputAddress(""+(j-1)));
		}
		return ordered;
	}

	/**
	 * More complex graph that has a fork in it
	 */
	@Test
	public void assignDepth_forked() {
		List<Node> ordered = createBranchA();

		SequenceForwardOrder alg = new SequenceForwardOrder((List)ordered);
		alg.assignDepth();

		assertEquals(0,alg.sequence.get(0).depth);
		assertEquals(1,alg.sequence.get(1).depth);
		assertEquals(1,alg.sequence.get(2).depth);
		assertEquals(2,alg.sequence.get(3).depth);
		assertEquals(3,alg.sequence.get(4).depth);
		assertEquals(4,alg.sequence.get(5).depth);
	}

	/**
	 * Graph with two paths, where one path is longer by two hops
	 */
	private List<Node> createBranchA() {
		List<Node> ordered = new ArrayList<>();

		ordered.add( create("0") );
		ordered.add( create("1") );
		ordered.add( create("2") );
		ordered.add( create("3") );
		ordered.add( create("4") );
		ordered.add( create("5") );

		ordered.get(1).sources.add( new InputAddress("0"));
		ordered.get(2).sources.add( new InputAddress("0"));
		ordered.get(3).sources.add( new InputAddress("1"));
		ordered.get(4).sources.add( new InputAddress("3"));
		ordered.get(5).sources.add( new InputAddress("4"));
		ordered.get(5).sources.add( new InputAddress("2"));
		return ordered;
	}

	/**
	 * See if it can find the first node in sequences of various lengths
	 */
	@Test
	public void findInput() {
		for (int i = 1; i < 4; i++) {
			List<Node> list = createLineSequence(i);
			Node first = list.get(0);
			Collections.shuffle(list,rand);

			SequenceForwardOrder alg = new SequenceForwardOrder((List)list);

			assertTrue(first == alg.findInput().node);
		}
	}

	private static void check( SequenceForwardOrder.NodeData n , String name , int next , int previous ) {
		assertTrue(name.equals(n.node.name));
		assertEquals(next,n.next.size());
		assertEquals(previous,n.previous.size());
	}

	private static void connected( SequenceForwardOrder.NodeData src , SequenceForwardOrder.NodeData dst ) {
		assertTrue(src.next.contains(dst));
		assertTrue(dst.previous.contains(src));
	}

	private static Node create( String name ) {
		Node n = new Node();
		n.name = name;
		return n;
	}
}
