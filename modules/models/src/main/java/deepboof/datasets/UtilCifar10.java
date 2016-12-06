package deepboof.datasets;

import deepboof.io.DeepBoofDataBaseOps;
import deepboof.misc.DeepBoofOps;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class UtilCifar10 {

	public static File downloadModelVggLike( File path ) {
		return DeepBoofDataBaseOps.downloadModel("http://heanet.dl.sourceforge.net/project/deepboof/networks/v1/likevgg_cifar10.zip", path);
	}

	public static File downloadData() {
		File trainingDir = DeepBoofOps.pathData("cifar10");

		// If needed, download required data sets and network model
		if( !trainingDir.exists() ) {
			try {
				DeepBoofDataBaseOps.download("http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz",trainingDir);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
			DeepBoofDataBaseOps.decompressTGZ(new File(trainingDir,"cifar-10-torch.tar.gz"),trainingDir);
			DeepBoofDataBaseOps.moveInsideAndDeleteDir(new File(trainingDir,"cifar-10-batches-t7"),trainingDir);
		}

		return trainingDir;
	}


	public static List<String> getClassNames() {
		List<String> names = new ArrayList<>();

		names.add("airplane");
		names.add("automobile");
		names.add("bird");
		names.add("cat");
		names.add("deer");
		names.add("dog");
		names.add("frog");
		names.add("horse");
		names.add("ship");
		names.add("truck");

		return names;
	}

}
