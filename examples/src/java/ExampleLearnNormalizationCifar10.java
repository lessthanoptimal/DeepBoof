import boofcv.alg.misc.ImageStatistics;
import boofcv.alg.misc.PixelMath;
import boofcv.core.image.border.BorderType;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel1D_F64;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;

import java.io.File;
import java.io.IOException;
import java.util.List;

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
		List<Planar<GrayF32>> listYuv = UtilCifar10.loadTrainingYuv(false).images;

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
		params.border = BorderType.ZERO.name();

		UtilCifar10.save(params,new File("YuvStatistics.txt"));
	}
}

