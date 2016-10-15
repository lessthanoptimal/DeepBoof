package deepboof;

import boofcv.alg.filter.stat.ImageLocalNormalization;
import boofcv.alg.misc.GImageMiscOps;
import boofcv.core.image.border.BorderType;
import boofcv.deepboof.DataManipulationOps;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.io.torch7.ConvertBoofToTorch;
import deepboof.io.torch7.SerializeBinaryTorch7;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchTensor;
import deepboof.models.DeepModelIO;
import deepboof.models.YuvStatistics;
import deepboof.tensors.Tensor_F32;
import deepboof.tensors.Tensor_U8;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Loads the previously computed input normalization parameters and applies it to the CIFAR10 training
 * data.  This is then saved to disk as a Torch object for training purposes later on.
 *
 * @author Peter Abeles
 */
//  For training sets
// TODO add other distortions
public class ExampleApplyNormalizeCifar10 {

	/**
	 * Apply the previously computed normalization to the input image set.  Also flip each
	 * input image horizontally to increase the diversity.
	 *
	 * Distortion: To generate more training data the original training data is modified by
	 *             applying horizontal flip.
	 *
	 * @param data
	 * @param stats
	 * @param distort If true it will generate additional data by distorting the original data set
	 * @param outputName
	 * @throws IOException
	 */
	private static void applyNormalization(DataSetsCifar10.DataSet data, YuvStatistics stats,
										   boolean distort ,
										   String outputName )
			throws IOException
	{
		// TODO do with a border of extend.  See how that changes score
		// ZERO border       = 72.03 at 69
		// EXTENDED border   = 70.14 at 81
		// NORMALIZED border = 70.95 at 147
		BorderType type = BorderType.valueOf(stats.border);
		ImageLocalNormalization<GrayF32> localNorm = new ImageLocalNormalization<>(GrayF32.class, type);
		Kernel1D_F32 kernel = DataManipulationOps.create1D_F32(stats.kernel);

		GrayF32 workspace = new GrayF32(32,32);

		int numOut = distort ? data.images.size()*2 : data.images.size();
		int step = distort ? 2 : 1;

		Tensor_F32 tensorYuv = new Tensor_F32(numOut,3,32,32);
		Tensor_U8 labelsOut = new Tensor_U8(numOut);

		for( int i = 0; i < data.images.size(); i++ ) {
			Planar<GrayF32> yuv = data.images.get(i);

			workspace.setTo(yuv.getBand(0));
			localNorm.zeroMeanStdOne(kernel,workspace,255.0,1e-4,yuv.getBand(0));
			DataManipulationOps.normalize(yuv.getBand(1), (float)stats.meanU, (float)stats.stdevU);
			DataManipulationOps.normalize(yuv.getBand(2), (float)stats.meanV, (float)stats.stdevV);

			DataManipulationOps.imageToTensor(yuv,tensorYuv,i*step);
			labelsOut.d[i*step] = data.labels.d[i];

			if( distort ) {
				// perform a horizontal
				GImageMiscOps.flipHorizontal(yuv);
				DataManipulationOps.imageToTensor(yuv, tensorYuv, i * step + 1);
				labelsOut.d[i * step + 1] = data.labels.d[i];
			}
		}

		System.out.println("Saving "+outputName+" to disk");
		SerializeBinaryTorch7 serializer = new SerializeBinaryTorch7(true);
		TorchTensor torchTensor = ConvertBoofToTorch.convert(tensorYuv);
		TorchTensor labels = ConvertBoofToTorch.convert(labelsOut);
		TorchGeneric torchMap = new TorchGeneric();
		torchMap.map.put("data",torchTensor);
		torchMap.map.put("label",labels);

		serializer.serialize(torchMap,new FileOutputStream(outputName));
	}

	public static void main(String[] args) throws IOException {

		// Load training data and convert into YUV image
		YuvStatistics stats = DeepModelIO.load(new File("YuvStatistics.txt"));

		System.out.println("Loading training data");
		DataSetsCifar10.DataSet data = DataSetsCifar10.loadTrainingYuv(false);
		applyNormalization(data, stats,true, "train_normalized_cifar10.t7");

		System.out.println("Loading test data");
		data = DataSetsCifar10.loadTestYuv(false);
		applyNormalization(data, stats,false, "test_normalized_cifar10.t7");

		System.out.println("   finished");
	}
}
