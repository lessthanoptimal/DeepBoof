package deepboof.datasets;

import deepboof.io.DeepBoofDataBaseOps;
import deepboof.misc.DeepBoofOps;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class UtilCifar10 {

	public static File downloadModelVggLike( File path ) {
//		if( !path.isDirectory() )
//			path = path.getParentFile();

		File pathToModel = new File(path,"likevgg_cifar10");
		if( !path.exists() ) {
			if (!path.mkdirs())
				throw new IllegalArgumentException("Failed to make path");
		} else {

			// check to see if the data already exists.  If so just return
			if( new File(pathToModel,"YuvStatistics.txt").exists() &&
					new File(pathToModel,"model.net").exists() )
				return pathToModel;

			// TODO check md5sum
		}

		System.out.println("Obtaining network model.  size = 125 MB");
		DeepBoofDataBaseOps.download("http://heanet.dl.sourceforge.net/project/deepboof/networks/v1/likevgg_cifar10.zip",path);
		DeepBoofDataBaseOps.decompressZip(new File(path,"likevgg_cifar10.zip"),path,true);

		return pathToModel;
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
