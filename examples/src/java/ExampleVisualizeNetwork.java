import boofcv.gui.image.ShowImages;
import deepboof.Function;
import deepboof.graph.FunctionSequence;
import deepboof.io.torch7.ParseBinaryTorch7;
import deepboof.io.torch7.SequenceAndParameters;
import deepboof.tensors.Tensor_F32;
import deepboof.visualization.SequentialNetworkDisplay;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class ExampleVisualizeNetwork {
	public static void main(String[] args) throws IOException {
		File modelHome = UtilCifar10.downloadModel();

		System.out.println("Load and convert to DeepBoof");
		SequenceAndParameters<Tensor_F32, Function<Tensor_F32>> sequence =
				new ParseBinaryTorch7().parseIntoBoof(new File(modelHome,"model.net"));

		FunctionSequence<Tensor_F32,Function<Tensor_F32>> network = sequence.createForward(3,32,32);

		SequentialNetworkDisplay gui = new SequentialNetworkDisplay((List)network.getSequence());

		JScrollPane scrollPane = new JScrollPane(gui);
		scrollPane.setPreferredSize(new Dimension(400,800));

		ShowImages.showWindow(scrollPane,"Network",true);
	}
}
