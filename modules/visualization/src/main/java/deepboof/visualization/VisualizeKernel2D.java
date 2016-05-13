package deepboof.visualization;

import deepboof.Tensor;

import javax.swing.*;
import java.awt.*;

/**
 * Renders a 2D kernel that's stored inside a tensor
 *
 *
 * @author Peter Abeles.
 */
public class VisualizeKernel2D extends JPanel {

	int kernel;
	int channel;
	int numRows,numCols;
	Tensor<?> tensor;

	boolean magnitude = false;

	public VisualizeKernel2D(int kernel , int channel , Tensor<?> tensor ) {
		this.kernel = kernel;
		this.channel = channel;
		this.tensor = tensor;

		this.numRows = tensor.length(2);
		this.numCols = tensor.length(3);
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D) g;

		int width = getWidth();
		int height = getHeight();

		double max = -Double.MAX_VALUE;
		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				max = Math.max(Math.abs(tensor.getDouble(kernel,channel,row,col)),max);
			}
		}

		int length = Math.max(numCols,numRows);

		for (int row = 0; row < numRows; row++) {
			int y0 = height*row/length;
			int y1 = height*(row+1)/length;

			for (int col = 0; col < numCols; col++) {
				int x0 = width*col/length;
				int x1 = width*(col+1)/length;

				double value = tensor.getDouble(kernel,channel,row,col)/max;

				Color color;
				if(magnitude) {
					int v = (int) (255 * Math.abs(value));
					color = new Color(v,v,v);
				} else {
					if (value > 0) {
						color = new Color((int) (255 * value), 0, 0);
					} else {
						value = -value;
						color = new Color(0, (int) (255 * value), 0);
					}
				}

				g2.setColor(color);
				g2.fillRect(x0,y0,x1-x0,y1-y0);
			}
		}
	}

	public void setShowMagnitude(boolean magnitude) {
		this.magnitude = magnitude;
	}
}
