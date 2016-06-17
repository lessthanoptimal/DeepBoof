import boofcv.alg.filter.stat.ImageLocalNormalization;
import boofcv.core.image.border.BorderType;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.Function;
import deepboof.graph.ForwardSequence;
import deepboof.io.torch7.ParseBinaryTorch7;
import deepboof.io.torch7.SequenceAndParameters;
import deepboof.misc.DataManipulationOps;
import deepboof.tensors.Tensor_F32;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static deepboof.misc.TensorOps.WI;

/**
 * Loads and tests a Torch model [1] which has been trained on the CIFAR-10 dataset [2].  This
 * example demonstrates how to load and process test data.  This includes the preprocessing step
 * of converting it into YUV color format and then normalizing the statistical properties of the
 * data to match how it was trained.  It is then sent through the network and performance evaluated.
 *
 * Around 92% test set accuracy is expected.
 *
 * [1] http://torch.ch/blog/2015/07/30/cifar.html
 * [2] http://www.cs.toronto.edu/~kriz/cifar.html
 *
 * @author Peter Abeles
 */
// TODO live update of confusion plot
public class ExampleClassifyCifar10TestSet {

	public static void main(String[] args) throws IOException {

		// Specify where all the prebuilt models and data sets are stored
		File modelHome = UtilCifar10.downloadModel();
		YuvStatistics stats = UtilCifar10.load(new File(modelHome,"YuvStatistics.txt"));

		System.out.println("Load and convert model to BoofCV");
		SequenceAndParameters<Tensor_F32, Function<Tensor_F32>> sequence =
				new ParseBinaryTorch7().parseIntoBoof(new File(modelHome,"model.net"));

		ForwardSequence<Tensor_F32,Function<Tensor_F32>> network = sequence.createForward(3,32,32);

		System.out.println("Loading test set data and converting into YUV");
		UtilCifar10.DataSet data = UtilCifar10.loadTestYuv(true);

		// Number of images in the test set
		int numTest = data.images.size();

		// Declare storage for the preprocessed input image and for the network's results
		Tensor_F32 tensorYuv = new Tensor_F32(1,3,32,32);
		Tensor_F32 output = new Tensor_F32(WI(1,network.getOutputShape()));
		// WI() is a convenience function which allows you to prepend another dimension onto the tensor's shape

		// Locally normalize using a gaussian kernel with zero padding
		BorderType type = BorderType.valueOf(stats.border);
		ImageLocalNormalization<GrayF32> localNorm = new ImageLocalNormalization<>(GrayF32.class, type);
		Kernel1D_F32 kernel = stats.create1D_F32();

		// Total number of correct guesses and number of guesses made
		int totalCorrect = 0;
		int totalConsidered = 1;

		List<String> classNames = UtilCifar10.getClassNames();

		// Currently estimated FPS and constant for how quickly the FPS estimate adapts
		double FPS = 0.0;
		double fpsFade = 0.85;

		// Main processing loop.   Classify and test the results of each image.
		for (int test = 0; test < numTest; test++) {
			long start = System.nanoTime();

			Planar<GrayF32> yuv = data.images.get(test);

			// Normalize the image
			localNorm.zeroMeanStdOne(kernel,yuv.getBand(0),255.0,1e-4,yuv.getBand(0));
			DataManipulationOps.normalize(yuv.getBand(1), (float)stats.meanU, (float)stats.stdevU);
			DataManipulationOps.normalize(yuv.getBand(2), (float)stats.meanV, (float)stats.stdevV);

			// Convert it from an image into a tensor
			DataManipulationOps.imageToTensor(yuv,tensorYuv,0);

			// Feed it through the CNN
			network.process(tensorYuv,output);

			FPS = fpsFade*FPS + (1.0-fpsFade)/((System.nanoTime()-start)*1e-9);

			// Select best fit and score the results
			double bestScore = -Double.MAX_VALUE;
			int bestType = -1;
			for (int i = 0; i < output.length(1); i++) {
				double score = output.get(0,i);
				if( score > bestScore ) {
					bestScore = score;
					bestType = i;
				}
			}

			String equality = "!=";
			if( data.labels.d[test] == bestType ) {
				equality = "==";
				totalCorrect++;
			}

			String found = classNames.get(bestType);
			String actual = classNames.get(data.labels.d[test]);

			System.out.printf("FPS = %5.2f Correct %6.2f%% out of %7d    %11s  %2s  %11s \n",
					FPS,100*(totalCorrect/(double)totalConsidered++),test+1,found,equality,actual);
		}
		System.out.println("Done!");
	}
}
