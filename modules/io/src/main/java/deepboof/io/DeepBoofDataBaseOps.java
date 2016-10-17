/*
 * Copyright (c) 2016, Peter Abeles. All Rights Reserved.
 *
 * This file is part of DeepBoof
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package deepboof.io;

import com.github.axet.wget.WGet;
import com.github.axet.wget.info.DownloadInfo;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;

/**
 * Functions for obtaining and processing network models
 *
 * @author Peter Abeles
 */
public class DeepBoofDataBaseOps {

	/**
	 * Attempts to download a model from the list of addresses.  If one fails it will try
	 * the next in the list.  After downloading the model it will then unzip it and return
	 * the path to the unzipped directory.  It is assumed the unzipped directory's name is
	 * the same as the file it's compressed in.
	 *
	 * @param addresses List of address that it can be downloaded from
	 * @param destination Directory that the file should be downloaded to and decompressed in
	 * @return The directory containing the decompressed
	 */
	public static File downloadModel(List<String> addresses , File destination ) {
		File pathDirectory = null;
		if( !destination.exists() )
			destination.mkdirs();

		for( String address : addresses ) {
			try {
				String fileName = new File(address).getName();
				pathDirectory = new File(destination, fileName.substring(0, fileName.length()-4));
				if( pathDirectory.exists() )
					break;
				DeepBoofDataBaseOps.download(address, destination);
				DeepBoofDataBaseOps.decompressZip(new File(destination, fileName), destination, true);
				break;
			} catch( RuntimeException e){
				pathDirectory = null;
			}
		}
		if( pathDirectory == null )
			throw new RuntimeException("Failed to download model");

		return pathDirectory;
	}

	/**
	 * Downloads a file from the specified address and saves it at the specified location
	 * @param src URL pointing to file
	 * @param dstDir destination of file
	 */
	public static File download( String src, File dstDir ) {
		try {
			File dst = new File(dstDir,new File(new URL(src).getFile()).getName());

			System.out.println("Downloading "+src+"  to "+dst);
			// make sure a file path exists
			if( !dstDir.exists() && !dstDir.mkdirs() )
				throw new RuntimeException("Can't create directories");
			DownloadInfo info = new DownloadInfo(new URL(src));
			info.extract();
			WGet w = new WGet(info, dst);
			w.download();
			return WGet.calcName(new URL(src),dst);
		} catch (MalformedURLException | RuntimeException e) {
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}

	public static void decompressTGZ( File src , File dst ) {
		Archiver archiver = ArchiverFactory.createArchiver("tar", "gz");
		try {
			archiver.extract(src, dst);
		} catch (IOException e) {
			System.out.println("Failed to decompress.  "+e.getMessage());
			System.exit(1);
		}
	}

	public static void decompressZip( File src , File dst , boolean deleteZip ) {
		try {
			ZipFile zipFile = new ZipFile(src);
			zipFile.extractAll(dst.getAbsolutePath());
			if( deleteZip ) {
				if( !src.delete() ) {
					System.err.println("Failed to delete "+src.getName());
				}
			}
		} catch (ZipException e) {
			System.err.println("Failed to decompress.  "+e.getMessage());
			System.exit(1);
		}
	}

	/**
	 * Moves everything that's inside the source directory into the destination directory then deletes the source dir.
	 * @param srcDir source directory
	 * @param dstDir destination directory
	 */
	public static void moveInsideAndDeleteDir(File srcDir , File dstDir ) {
		if( !srcDir.isDirectory() ) {
			System.out.println(srcDir.getName()+" isn't a directory");
			System.exit(1);
		}
		if( !dstDir.isDirectory() ) {
			System.out.println(dstDir.getName()+" isn't a directory");
			System.exit(1);
		}

		for( File f : srcDir.listFiles() ) {
			try {
				Files.move(f.toPath(),new File(dstDir,f.getName()).toPath(), StandardCopyOption.REPLACE_EXISTING);
			} catch (IOException e) {
				System.out.println("Failed to move.  "+e.getMessage());
				System.exit(1);
			}
		}
		if( !srcDir.delete() ) {
			System.err.println("Failed to cleanup "+srcDir.getName());
		}
	}
}
