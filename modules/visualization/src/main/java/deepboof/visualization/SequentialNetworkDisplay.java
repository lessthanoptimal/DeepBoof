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

package deepboof.visualization;

import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.util.mxConstants;
import com.mxgraph.view.mxGraph;
import deepboof.Tensor;
import deepboof.forward.*;
import deepboof.graph.Node;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Peter Abeles
 */
// TODO click on node to show configuration
	// TODO visualize data of selected
public class SequentialNetworkDisplay extends JPanel {

	mxGraphComponent graphComponent;
	mxGraph graph;
	Map<String,Node> nameToNode = new HashMap<>();
	JTextPaneAA textConfig = new JTextPaneAA();

	JPanel visualizePanel = new JPanel();

	public SequentialNetworkDisplay( List<Node<?,?>> list ) {
		setLayout(new BorderLayout());
		graph = new mxGraph();
		Object parent = graph.getDefaultParent();

		graph.getModel().beginUpdate();
		try
		{
			int nodeWidth = 140;
			int nodeHeight = 35;

			int y = 0;

			Node<?,?> prev = list.get(0);
			Object prevVertex = graph.insertVertex(parent, null, getTitle(prev),20, y, nodeWidth,nodeHeight);
			nameToNode.put(getTitle(prev),prev);
			setCellColor(prev,prevVertex);
			y += nodeHeight+20;

			for (int i = 1; i < list.size(); i++) {
				Node<?,?> curr = list.get(i);
				String title = getTitle(curr);
				Object currVertex = graph.insertVertex(parent, null, title, 20, y, nodeWidth,nodeHeight);
				nameToNode.put(title,curr);
				setCellColor(curr,currVertex);

				graph.insertEdge(parent, null, null, prevVertex, currVertex);


				prev = curr;
				prevVertex = currVertex;

				y += nodeHeight+20;
			}
		}
		finally
		{
			graph.getModel().endUpdate();
		}

		JPanel right = new JPanel();
		right.setLayout(new BoxLayout(right, BoxLayout.Y_AXIS));

		textConfig.setMinimumSize(new Dimension(200,300));
		textConfig.setPreferredSize(new Dimension(200,300));
		Font font = new Font("monospaced", Font.PLAIN, 12);
		textConfig.setFont(font);

//		visualizePanel.setMinimumSize(new Dimension(400,400));
//		visualizePanel.setPreferredSize(new Dimension(400,400));

		right.add(textConfig);
		right.add(visualizePanel);

		graphComponent = new mxGraphComponent(graph);
		add(graphComponent,BorderLayout.CENTER);
		add(right,BorderLayout.EAST);

		graphComponent.getGraphControl().addMouseListener(new ClickHandler());
	}

	private void setCellColor( Node node , Object cell ) {

		String color = "DDDAFF";

		if( SpatialConvolve2D.class.isAssignableFrom(node.function.getClass())) {
			color = "#DDDABB";
		} else if( SpatialBatchNorm.class.isAssignableFrom(node.function.getClass()) ||
				FunctionBatchNorm.class.isAssignableFrom(node.function.getClass())
				) {
			color = "#EEFABB";
		} else if( FunctionLinear.class.isAssignableFrom(node.function.getClass())) {
			color = "#FFDADD";
		} else if( ActivationReLU.class.isAssignableFrom(node.function.getClass())) {
			color = "#DAFFDD";
		}

		graph.setCellStyles(mxConstants.STYLE_FILLCOLOR, color, new Object[]{cell});

	}

	private String getTitle(Node<?, ?> curr) {
		return curr.function.getClass().getSimpleName()+"\n"+curr.name;
	}

	private class ClickHandler extends MouseAdapter {
		@Override
		public void mousePressed(MouseEvent e) {
			Object cell = graphComponent.getCellAt(e.getX(), e.getY());

				if (cell != null)
				{
					String label = graph.getLabel(cell);
					System.out.println("cell="+graph.getLabel(cell));

					Node n = nameToNode.get(label);
					if( n == null )
						return;

					String info = "";
					visualizePanel.removeAll();
					if( n.function instanceof SpatialConvolve2D ) {
						info += configString((SpatialConvolve2D)n.function);


						List parameters = n.function.getParameters();
						if( parameters != null ) {
							Kernel2DGridPanel vis = new Kernel2DGridPanel((Tensor)parameters.get(0),60,60);
							visualizePanel.add(vis);
						}
					} else if( n.function instanceof FunctionBatchNorm ) {
						info += configString((FunctionBatchNorm)n.function);
					} else if( n.function instanceof SpatialMaxPooling ) {
						info += configString((SpatialMaxPooling)n.function);
					}
					visualizePanel.invalidate();

					final String finfo = info;

					SwingUtilities.invokeLater(new Runnable() {
						@Override
						public void run() {
							textConfig.setText(finfo);
						}
					});
				}
		}
	}

	private String configString( SpatialConvolve2D operator ) {
		ConfigConvolve2D config = operator.getConfiguration();

		String out = "";
		out += String.format("Number of kernels   %d\n",config.getTotalKernels());
		out += configString(config);
		out += configString(operator.getPadding());

		return out;
	}

	private String configString( FunctionBatchNorm operator ) {

		String out = "";
		out += "gamma-beta          "+operator.hasGammaBeta()+"\n";
		return out;
	}

	private String configString( SpatialMaxPooling operator ) {

		String out = "";
		out += configString(operator.getConfiguration());
		return out;
	}

	private String configString( ConfigSpatial config ) {

		String out = "";
		out += String.format("Period X            %d\n",config.periodX);
		out += String.format("       Y            %d\n",config.periodY);
		out += String.format("Window width        %d\n",config.HH);
		out += String.format("       height       %d\n",config.WW);
		return out;
	}

	private String configString( SpatialPadding2D padding ) {

		String out = "\n";
		out += padding.getClass().getSimpleName()+"\n";

		if( padding instanceof ConstantPadding2D) {
			ConstantPadding2D p = (ConstantPadding2D)padding;
			out += String.format("   value         %6.2f\n",p.getPaddingValue());
		}

		out += String.format("      x0           %2d\n",padding.getPaddingCol0());
		out += String.format("      y0           %2d\n",padding.getPaddingRow0());
		out += String.format("      x1           %2d\n",padding.getPaddingCol1());
		out += String.format("      y1           %2d\n",padding.getPaddingRow1());
		return out;
	}
}
