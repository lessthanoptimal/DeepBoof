import boofcv.alg.misc.ImageStatistics;
import boofcv.alg.misc.PixelMath;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel1D_F64;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.io.torch7.ParseAsciiTorch7;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;

/**
 * Computes statistics across the entire input data set so that it can be normalized to ensure
 * that the inputs have a mean of 0 and standard deviation of 1.
 *
 * Steps:
 * 1) This converts the input image from RGB into YUV color.
 * 2) Then computes the global mean and standard deviation for U and V bands, these are the color bands.
 * 3) The Y (gray scale) band will be normalized later on using a local spatial algorithm, but the
 *    Gaussian kernel used for that future normalization is saved to disk to ensure repeatability..
 *
 * @author Peter Abeles
 */
public class ExampleLearnNormalizationCifar10 {

	public static void main(String[] args) throws IOException {

		// Load training data and convert into YUV image
		File trainingDir = UtilCifar10.downloadData();
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();

		System.out.println("Loading images");
		List<Planar<GrayF32>> listYuv = new ArrayList<>();
		for( File f : trainingDir.listFiles() ) {
			if( !f.getName().startsWith("data_"))
				continue;

			Map<Object,TorchObject> map = ((TorchGeneric)ascii.parseOne(f)).map;
			listYuv.addAll(UtilCifar10.convertToYuv(convert(map.get("data")),false));
		}

		// Compute mean and standard deviation for U and V bands
		System.out.println("Computing mean");
		int totalPixels = listYuv.size()*32*32;
		double meanU = 0;
		double meanV = 0;

		for( Planar<GrayF32> yuv : listYuv ) {
			meanU += ImageStatistics.sum(yuv.getBand(1));
			meanV += ImageStatistics.sum(yuv.getBand(2));
		}
		meanU /= totalPixels;
		meanV /= totalPixels;

		// compute standard deviation using Sum(x[i]^2) - n*mean(x)^2
		System.out.println("Computing standard deviation");
		double stdevU = 0;
		double stdevV = 0;

		for( Planar<GrayF32> yuv : listYuv ) {
			for (int i = 1; i < 3; i++) {
				PixelMath.pow2(yuv.getBand(i),yuv.getBand(i));
			}
			stdevU += ImageStatistics.sum(yuv.getBand(1));
			stdevV += ImageStatistics.sum(yuv.getBand(2));
		}

		stdevU = Math.sqrt( stdevU/totalPixels - meanU*meanU );
		stdevV = Math.sqrt( stdevV/totalPixels - meanV*meanV );

		// smoothing kernel used in spatial normalization in Y channel
		Kernel1D_F64 kernel = FactoryKernelGaussian.gaussian(Kernel1D_F64.class,-1,4);

		// Save these statistics
		System.out.println("Saving");
		YuvStatistics params = new YuvStatistics();
		params.meanU = meanU;
		params.meanV = meanV;
		params.stdevU = stdevU;
		params.stdevV = stdevV;
		params.kernel = kernel.data;

		UtilCifar10.save(params,new File("YuvStatistics.txt"));
	}
}

