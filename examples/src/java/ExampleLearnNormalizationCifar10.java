import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.io.DatabaseOps;
import deepboof.io.torch7.ParseAsciiTorch7;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchObject;
import deepboof.misc.DeepBoofOps;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;

/**
 * @author Peter Abeles
 */
public class ExampleLearnNormalizationCifar10 {
	public static int width = 32;
	public static int height = 32;

	public static void main(String[] args) throws IOException {
		File trainingDir = DeepBoofOps.pathData("cifar10");

				// If needed, download required data sets and network model
		if( !trainingDir.exists() ) {
			System.out.println("Obtaining training and testing data. size = 175 MB");
			DatabaseOps.download("http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz",trainingDir);
			DatabaseOps.decompressTGZ(new File(trainingDir,"cifar-10-torch.tar.gz"),trainingDir);
			DatabaseOps.moveInsideAndDeleteDir(new File(trainingDir,"cifar-10-batches-t7"),trainingDir);
		}

		// Compute the average for U and V bands
		Planar<GrayF32> sum = new Planar<>(GrayF32.class,width,height,3);
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


		// Apply spatial normalization to Y and global to U and V

		// Save the results
	}
}

