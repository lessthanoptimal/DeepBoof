import boofcv.alg.color.ColorYuv;
import boofcv.core.image.GConvertImage;
import boofcv.gui.image.ImageGridPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.Planar;
import deepboof.io.DatabaseOps;
import deepboof.misc.DeepBoofOps;
import deepboof.tensors.Tensor_U8;

import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class UtilCifar10 {

	public static void save(YuvStatistics params , File file ) throws FileNotFoundException {
		PrintStream out = new PrintStream(file);

		out.printf("meanU %f\n",params.meanU);
		out.printf("meanV %f\n",params.meanV);
		out.printf("stdevU %f\n",params.stdevU);
		out.printf("stdevV %f\n",params.stdevV);
		out.print("kernel");
		for (int i = 0; i < params.kernel.length; i++) {
			out.printf(" %.10f",params.kernel[i]);
		}
		out.println();
		out.close();
	}

	public static YuvStatistics load(File file ) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));

		YuvStatistics out = new YuvStatistics();
		out.meanU = readDouble(reader.readLine());
		out.meanV = readDouble(reader.readLine());
		out.stdevU = readDouble(reader.readLine());
		out.stdevV = readDouble(reader.readLine());
		out.kernel = readArray(reader.readLine());

		return out;
	}

	private static double readDouble( String line ) {
		return Double.parseDouble(line.split(" ")[1]);
	}

	private static double[] readArray( String line ) {
		String words[] = line.split(" ");
		double[] out = new double[ words.length-1 ];
		for (int i = 0; i < out.length; i++) {
			out[i] = Double.parseDouble(words[i+1]);
		}
		return out;
	}

	public static File downloadModel() {
		File modelHome = DeepBoofOps.pathData("torch_models/likevgg_cifar10");

		if( !modelHome.exists() ) {
			System.out.println("Obtaining network model.  size = 125 MB");
			File modelParent = modelHome.getParentFile();
			DatabaseOps.download("http://heanet.dl.sourceforge.net/project/deepboof/networks/v1/likevgg_cifar10.zip",modelParent);
			DatabaseOps.decompressZip(new File(modelParent,"likevgg_cifar10.zip"),modelParent,true);
		}

		return modelHome;
	}

	public static File downloadData() {
		File trainingDir = DeepBoofOps.pathData("cifar10");

		// If needed, download required data sets and network model
		if( !trainingDir.exists() ) {
			System.out.println("Obtaining training and testing data. size = 175 MB");
			DatabaseOps.download("http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz",trainingDir);
			DatabaseOps.decompressTGZ(new File(trainingDir,"cifar-10-torch.tar.gz"),trainingDir);
			DatabaseOps.moveInsideAndDeleteDir(new File(trainingDir,"cifar-10-batches-t7"),trainingDir);
		}

		return trainingDir;
	}

	/**
	 * The input test set tensor is stored in a weird format.  This will unroll each
	 * image and convert it into a YUV image.
	 */
	public static List<Planar<GrayF32>> convertToYuv(Tensor_U8 raw , boolean showFirstImages ) {

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
