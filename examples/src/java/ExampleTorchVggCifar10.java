import boofcv.alg.color.ColorYuv;
import boofcv.alg.filter.stat.ImageLocalNormalization;
import boofcv.core.image.GConvertImage;
import boofcv.core.image.border.BorderType;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.gui.image.ImageGridPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.Planar;
import deepboof.Function;
import deepboof.graph.ForwardSequence;
import deepboof.io.DatabaseOps;
import deepboof.io.torch7.ParseAsciiTorch7;
import deepboof.io.torch7.ParseBinaryTorch7;
import deepboof.io.torch7.SequenceAndParameters;
import deepboof.io.torch7.TorchUtilities;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchObject;
import deepboof.misc.DataManipulationOps;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_U8;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;
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
public class ExampleTorchVggCifar10 {

	public static void main(String[] args) throws IOException {

		// Specify where all the prebuilt models and data sets are stored
		File modelHome = new File("data/torch_models/likevgg_cifar10");
		File inputFile = new File("data/cifar10/test_batch.t7");

		// If needed, download required data sets and network model
		if( !inputFile.exists() ) {
			System.out.println("Obtaining training and testing data. size = 175 MB");
			File inputHome = inputFile.getParentFile();
			DatabaseOps.download("http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz",inputHome);
			DatabaseOps.decompressTGZ(new File(inputHome,"cifar-10-torch.tar.gz"),inputHome);
			DatabaseOps.moveInsideAndDeleteDir(new File(inputHome,"cifar-10-batches-t7"),inputHome);
		}

		if( !modelHome.exists() ) {
			System.out.println("Obtaining network model.  size = 125 MB");
			File modelParent = modelHome.getParentFile();
			DatabaseOps.download("http://heanet.dl.sourceforge.net/project/deepboof/networks/v1/likevgg_cifar10.zip",modelParent);
			DatabaseOps.decompressZip(new File(modelParent,"likevgg_cifar10.zip"),modelParent,true);
			System.out.println("Testing the network");
			TorchUtilities.validateNetwork(modelHome,true);
		}

		System.out.println("Load and convert to BoofCV");
		SequenceAndParameters<Tensor_F32, Function<Tensor_F32>> sequence =
				new ParseBinaryTorch7().parseIntoBoof(new File(modelHome,"model.net"));

		ForwardSequence<Tensor_F32,Function<Tensor_F32>> network = sequence.createForward(3,32,32);

		System.out.println("Loading evaluation data");
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();
		Map<Object,TorchObject> testMap = ((TorchGeneric)ascii.parseOne(inputFile)).map;

		// This file describes how to normalize part of the input image
		TorchGeneric normParam = ascii.parseOne(new File(modelHome,"normalization_parameters.t7"));

		float mean_u = (float)normParam.getNumber("mean_u");
		float mean_v = (float)normParam.getNumber("mean_u");
		float std_u = (float)normParam.getNumber("std_u");
		float std_v = (float)normParam.getNumber("std_v");

		// Ground truth labels for each of the images
		Tensor_U8 labels = convert(testMap.get("labels"));

		// Convert the input RGB images into YUV color space and (optionally) display a few of them
		List<Planar<GrayF32>> listTestYuv = convertToYuv(convert(testMap.get("data")),true);

		// Number of images in the test set
		int numTest = listTestYuv.size();

		// Declare storage for the preprocessed input image and for the network's results
		Tensor_F32 tensorYuv = new Tensor_F32(1,3,32,32);
		Tensor_F32 output = new Tensor_F32(WI(1,network.getOutputShape()));
		// WI() is a convinience function which allows you to prepend another dimension onto the tensor's shape

		// Locally normalize using a gaussian kernel with zero padding
		ImageLocalNormalization<GrayF32> localNorm = new ImageLocalNormalization<>(GrayF32.class, BorderType.ZERO);
		// TODO This is probably not the same kernel it was processed with
		Kernel1D_F32 kernel = FactoryKernelGaussian.gaussian(1,true,32,-1,3);

		// Total number of correct guesses and number of guesses made
		int totalCorrect = 0;
		int totalConsidered = 1;

		// Currently estimated FPS and constant for how quickly the FPS estimate adapts
		double FPS = 0.0;
		double fpsFade = 0.85;

		// Main processing loop.   Classify and test the results of each image.
		for (int test = 0; test < numTest; test++) {
			long start = System.nanoTime();

			Planar<GrayF32> yuv = listTestYuv.get(test);

			// Normalize the image
			localNorm.zeroMeanStdOne(kernel,yuv.getBand(0),255.0,1e-4,yuv.getBand(0));
			DataManipulationOps.normalize(yuv.getBand(1), mean_u, std_u);
			DataManipulationOps.normalize(yuv.getBand(2), mean_v, std_v);

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

			if( labels.d[test] == bestType ) {
				totalCorrect++;
			}

			System.out.printf("FPS = %5.2f Selected %2d correct = %6.2f at index %5d\n",FPS,bestType,(totalCorrect/(double)totalConsidered++),test);
		}
		System.out.println("Done!");
	}

	/**
	 * The input test set tensor is stored in a weird format.  This will unroll each
	 * image and convert it into a YUV image.
	 */
	public static List<Planar<GrayF32>> convertToYuv( Tensor_U8 raw , boolean showFirstImages ) {

		List<Planar<GrayF32>> output = new ArrayList<>();

		int numTest = raw.length(1);

		Planar<GrayU8> color = new Planar<>(GrayU8.class,32,32,3);
		Planar<GrayF32> rgb = new Planar<>(GrayF32.class,32,32,3);

		ImageGridPanel gui = null;
		if( showFirstImages ) {
			gui = new ImageGridPanel(6,6);
		}

		for (int test = 0; test < numTest; test++) {
			// parse the ass backwards RGB format that the test image was stored in
			int imageIndex = 0;
			for (int band = 0; band < 3; band++) {
				GrayU8 gray = color.getBand(band);
				for (int y = 0; y < 32; y++) {
					for (int x = 0; x < 32; x++, imageIndex++) {
						gray.data[y * 32 + x] = raw.d[raw.idx(imageIndex, test)];
					}
				}
			}

			if( showFirstImages && test < 36) {
				BufferedImage buffered = ConvertBufferedImage.convertTo(rgb,null,true);
				gui.setImage(test/6,test%6,buffered);
			}

			Planar<GrayF32> yuv = new Planar<>(GrayF32.class,32,32,3);
			GConvertImage.convert(color,rgb);
			ColorYuv.rgbToYuv_F32(rgb,yuv);

			output.add( yuv );
		}

		if( showFirstImages ) {
			gui.autoSetPreferredSize();
			ShowImages.showWindow(gui,"CIFAR-10 Images",true);
		}

		return output;
	}

}