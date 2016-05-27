import boofcv.alg.filter.stat.ImageLocalNormalization;
import boofcv.core.image.border.BorderType;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import deepboof.io.torch7.ConvertBoofToTorch;
import deepboof.io.torch7.ParseAsciiTorch7;
import deepboof.io.torch7.SerializeBinaryTorch7;
import deepboof.io.torch7.struct.TorchGeneric;
import deepboof.io.torch7.struct.TorchObject;
import deepboof.io.torch7.struct.TorchTensor;
import deepboof.misc.DataManipulationOps;
import deepboof.tensors.Tensor_F32;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static deepboof.io.torch7.ConvertTorchToBoofForward.convert;

/**
 * Loads the previously computed input normalization parameters and applies it to the CIFAR10 training
 * data.  This is then saved to disk as a Torch object for training purposes later on.
 *
 * @author Peter Abeles
 */
public class ExampleApplyNormalizeCifar10 {
	public static void main(String[] args) throws IOException {

		// Load training data and convert into YUV image
		System.out.println("Loading images");
		File trainingDir = UtilCifar10.downloadData();
		ParseAsciiTorch7 ascii = new ParseAsciiTorch7();

		List<Planar<GrayF32>> listYuv = new ArrayList<>();
		for( File f : trainingDir.listFiles() ) {
			if( !f.getName().startsWith("data_"))
				continue;

			Map<Object,TorchObject> map = ((TorchGeneric)ascii.parseOne(f)).map;
			listYuv.addAll(UtilCifar10.convertToYuv(convert(map.get("data")),false));
		}

		System.out.println("Normalizing");
		YuvStatistics stats = UtilCifar10.load(new File("YuvStatistics.txt"));

		ImageLocalNormalization<GrayF32> localNorm = new ImageLocalNormalization<>(GrayF32.class, BorderType.ZERO);
		Kernel1D_F32 kernel = stats.create1D_F32();

		GrayF32 workspace = new GrayF32(32,32);

		Tensor_F32 tensorYuv = new Tensor_F32(listYuv.size(),3,32,32);

		for( int i = 0; i < listYuv.size(); i++ ) {
			Planar<GrayF32> yuv = listYuv.get(i);

			workspace.setTo(yuv.getBand(0));
			localNorm.zeroMeanStdOne(kernel,workspace,255.0,1e-4,yuv.getBand(0));
			DataManipulationOps.normalize(yuv.getBand(1), (float)stats.meanU, (float)stats.stdevU);
			DataManipulationOps.normalize(yuv.getBand(2), (float)stats.meanV, (float)stats.stdevV);

			DataManipulationOps.imageToTensor(yuv,tensorYuv,i);
		}

		System.out.println("Saving to disk");
		SerializeBinaryTorch7 serializer = new SerializeBinaryTorch7(true);
		TorchTensor torchTensor = ConvertBoofToTorch.convert(tensorYuv);
		serializer.serialize(torchTensor,new FileOutputStream("train_normalized_cifar10.t7"));

		System.out.println("   finished");
	}
}
