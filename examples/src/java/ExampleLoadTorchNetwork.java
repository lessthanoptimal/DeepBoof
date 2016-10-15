import deepboof.io.torch7.ParseBinaryTorch7;

import java.io.File;
import java.io.IOException;

/**
 * @author Peter Abeles
 */
// https://gist.github.com/szagoruyko/0f5b4c5e2d2b18472854
// https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/ninbn.lua
public class ExampleLoadTorchNetwork {
	public static void main(String[] args) throws IOException {
		String path = "nin_bn_final.t7";

		Object o = new ParseBinaryTorch7().setVerbose(true).
				parseIntoBoof(new File(path));

		System.out.println("Done");
	}
}
