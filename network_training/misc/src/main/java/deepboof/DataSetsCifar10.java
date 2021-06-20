package deepboof;

import boofcv.alg.color.ColorYuv;
import boofcv.core.image.GConvertImage;
import boofcv.gui.image.ImageGridPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.Planar;
import deepboof.io.DeepBoofDataBaseOps;
import deepboof.io.torch7.ParseAsciiTorch7;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchObject;
import deepboof.misc.DeepBoofOps;
import deepboof.tensors.Tensor_U8;
import org.ddogleg.struct.DogArray_I8;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;

/**
 * @author Peter Abeles
 */
public class DataSetsCifar10 {

	public static DataSet loadTrainingYuv(boolean show) throws IOException {
		File trainingDir = DataSetsCifar10.downloadData();
		System.out.println("Training data set "+trainingDir.getPath());
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();

		List<Planar<GrayF32>> listYuv = new ArrayList<>();
		DogArray_I8 labels = new DogArray_I8();
		for( File f : trainingDir.listFiles() ) {
			if( !f.getName().startsWith("data_"))
				continue;

			Map<Object,TorchObject> map = ((TorchGeneric)ascii.parseOne(f)).map;
			listYuv.addAll(DataSetsCifar10.convertToYuv(convert(map.get("data")),show));

			Tensor_U8 l = convert(map.get("labels"));
			labels.addAll(l.d,0,l.d.length);
		}
		byte d[] = new byte[labels.size()];
		System.arraycopy(labels.data,0,d,0,labels.size());
		labels.data = d;

		Tensor_U8 l = Tensor_U8.wrap(labels.data,labels.size());

		return new DataSet(listYuv,l);
	}

	public static DataSet loadTestYuv(boolean show) throws IOException {
		File trainingDir = DataSetsCifar10.downloadData();
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();

		File f = new File(trainingDir,"test_batch.t7");

		Map<Object,TorchObject> map = ((TorchGeneric)ascii.parseOne(f)).map;
		List<Planar<GrayF32>> listYuv =  DataSetsCifar10.convertToYuv(convert(map.get("data")),show);
		Tensor_U8 labels = convert(map.get("labels"));
		labels.reshape(labels.length());// it's saved with a weird shape that messes stuff up

		return new DataSet(listYuv,labels);
	}

	public static File downloadData() {
		File trainingDir = DeepBoofOps.pathData("cifar10");

		// If needed, download required data sets and network model
		if( !trainingDir.exists() ) {
			System.out.println("Obtaining training and testing data. size = 175 MB");
			DeepBoofDataBaseOps.download("http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz",trainingDir);
			DeepBoofDataBaseOps.decompressTGZ(new File(trainingDir,"cifar-10-torch.tar.gz"),trainingDir);
			DeepBoofDataBaseOps.moveInsideAndDeleteDir(new File(trainingDir,"cifar-10-batches-t7"),trainingDir);
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
			ShowImages.showWindow(gui,"CIFAR-10 Images",false);
		}

		return output;
	}

	public static class DataSet {
		public List<Planar<GrayF32>> images;
		public Tensor_U8 labels;

		public DataSet(List<Planar<GrayF32>> images, Tensor_U8 labels) {
			this.images = images;
			this.labels = labels;
		}

		public DataSet() {
		}
	}
}
