package deepboof.datasets;

import deepboof.io.DatabaseOps;
import deepboof.misc.DeepBoofOps;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class UtilCifar10 {
	public static File downloadModelVggLike() {
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
