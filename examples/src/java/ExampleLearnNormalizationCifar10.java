import boofcv.alg.misc.ImageStatistics;
import boofcv.alg.misc.PixelMath;
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
 * Computes statistics across the input data set and saves the found parameters.  Networks train better when inputs
 * have been normalizes such that they are between -1 and 1.  This converts the input image from RGB into YUV color.
 * Then computes the global mean and standard deviation for U and V bands, these are the color bands.  Then
 * will perform a local gaussian weighted normalization in the Y (gray scale) band.
 *
 * @author Peter Abeles
 */
public class ExampleLearnNormalizationCifar10 {

	public static void main(String[] args) throws IOException {
		File trainingDir = UtilCifar10.downloadData();

		// Compute the average for U and V bands
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();

		// Load training data and convert into YUV image
		List<Planar<GrayF32>> listYuv = new ArrayList<>();
		for( File f : trainingDir.listFiles() ) {
			if( !f.getName().startsWith("data_"))
				continue;

			Map<Object,TorchObject> map = ((TorchGeneric)ascii.parseOne(f)).map;
			listYuv.addAll(UtilCifar10.convertToYuv(convert(map.get("data")),false));
		}

		// Compute mean and standard deviation for U and V bands
		double meanU = 0;
		double meanV = 0;

		for( Planar<GrayF32> yuv : listYuv ) {
			meanU += ImageStatistics.sum(yuv.getBand(1));
			meanV += ImageStatistics.sum(yuv.getBand(2));
		}
		meanU /= listYuv.size();
		meanV /= listYuv.size();

		// compute standard deviation using Sum(x[i]^2) - n*mean(x)^2
		double stdevU = 0;
		double stdevV = 0;

		for( Planar<GrayF32> yuv : listYuv ) {
			for (int i = 1; i < 3; i++) {
				PixelMath.pow2(yuv.getBand(i),yuv.getBand(i));
			}
			stdevU += ImageStatistics.sum(yuv.getBand(1));
			stdevV += ImageStatistics.sum(yuv.getBand(2));
		}

		stdevU = Math.sqrt( stdevU - meanU*meanU*listYuv.size());
		stdevV = Math.sqrt( stdevV - meanV*meanV*listYuv.size());

		// Save these statistics

		// Spatial normalization on Y will be done using a Gaussian kernel.  Save the exact kernel to ensure
		// its reproducible

	}
}

