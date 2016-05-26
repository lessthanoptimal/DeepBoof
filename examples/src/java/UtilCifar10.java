import boofcv.alg.color.ColorYuv;
import boofcv.core.image.GConvertImage;
import boofcv.gui.image.ImageGridPanel;
import boofcv.gui.image.ShowImages;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.Planar;
import deepboof.tensors.Tensor_U8;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class UtilCifar10 {
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
