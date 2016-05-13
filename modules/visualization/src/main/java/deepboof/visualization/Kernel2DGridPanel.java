package deepboof.visualization;

import deepboof.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Peter Abeles.
 */
public class Kernel2DGridPanel extends JPanel implements ActionListener {

	JCheckBox toggleMagnitude = new JCheckBox("Magnitude");


	List<VisualizeKernel2D> widgets = new ArrayList<>();

	public Kernel2DGridPanel( Tensor<?> kernelWeights , int kernelWidth , int kernelHeight  ) {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

		toggleMagnitude.setSelected(true);
		toggleMagnitude.addActionListener(this);

		int numKernels = kernelWeights.length(0);
		int numChannels = kernelWeights.length(1);

		JPanel gridPanel = new JPanel();
		gridPanel.setLayout(new GridLayout(numKernels,numChannels,10,10));

		for (int kernel = 0; kernel < numKernels; kernel++) {
			for (int channel = 0; channel < numChannels; channel++) {
				JPanel foo = new JPanel();
				foo.setLayout(new BoxLayout(foo, BoxLayout.Y_AXIS));

				VisualizeKernel2D widget = new VisualizeKernel2D(kernel,channel,kernelWeights);
				widget.setShowMagnitude(true);
				widgets.add(widget);

				widget.setPreferredSize(new Dimension(kernelWidth,kernelHeight));
				widget.setMinimumSize(new Dimension(kernelWidth,kernelHeight));
				widget.setMaximumSize(new Dimension(kernelWidth,kernelHeight));

				foo.add( new JLabel(String.format("%3d %3d",kernel,channel)));
				foo.add(widget);

				gridPanel.add( foo);
			}
		}

		add(toggleMagnitude);
		add(gridPanel);

	}

	@Override
	public void actionPerformed(ActionEvent e) {
		boolean magnitude = toggleMagnitude.isSelected();

		for( VisualizeKernel2D widget : widgets ) {
			widget.setShowMagnitude(magnitude);
		}
		repaint();
	}
}
