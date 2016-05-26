package deepboof.misc;

import java.io.File;

/**
 * @author Peter Abeles
 */
public class DeepBoofOps {
	public static File pathData( String relativePath ) {
		return new File(pathToBase(),new File("data",relativePath).getPath());
	}

	public static File pathToBase() {
		File path = new File(".").getAbsoluteFile();

		while( true ) {
			File[] children = path.listFiles();

			boolean foundModules = false;
			boolean foundData = false;
			boolean foundSettings = false;

			for( File c : children ) {
				if( c.getName().equals("modules"))
					foundModules = true;
				else if( c.getName().equals("data"))
					foundData = true;
				else if( c.getName().equals("settings.gradle"))
					foundSettings = true;
			}

			if( foundModules && foundData && foundSettings )
				return path;
			path = path.getParentFile();
			if( path == null )
				throw new RuntimeException("Can't find base of project DeepBoof!  Run from inside");
		}
	}
}
