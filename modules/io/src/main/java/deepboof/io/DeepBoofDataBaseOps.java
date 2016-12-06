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

import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.rauschig.jarchivelib.Archiver;
import org.rauschig.jarchivelib.ArchiverFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
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
	 * @return The directory containing the decompressed model
	 */
	public static File downloadModel(List<String> addresses , File destination ) {
		if( addresses.size() == 0 )
			return null;

		if( !destination.exists() )
			if( !destination.mkdirs() ) {
				throw new RuntimeException("Can't create destination directories");
			}

		// see if the decompressed directory already exists.  If it does skip downloading
		String fileName = new File(addresses.get(0)).getName();
		File pathDirectory = new File(destination, fileName.substring(0, fileName.length()-4));
		if( pathDirectory.exists() ) {
			return pathDirectory;
		}

		// download the file
		int which = download(addresses, new File(destination,fileName) );
		if( which >= 0 ) {
			DeepBoofDataBaseOps.decompressZip(new File(destination, fileName), destination, true);
			return pathDirectory;
		} else {
			throw new RuntimeException("Failed to download model");
		}
	}

	/**
	 * @see #downloadModel(List, File)
	 *
	 * @param address Address to downloaded from
	 * @param destination Directory that the file should be downloaded to and decompressed in
	 * @return The directory containing the decompressed model
	 */
	public static File downloadModel(String address , File destination ) {
		List<String> addresses = new ArrayList<>();
		addresses.add( address );
		return downloadModel(addresses, destination);
	}

	/**
	 * Will attempt to download the file from the list of URLs.  If one failes it will go to the next in the
	 * list after printing why it failed
	 * @param urls Sources for the file
	 * @param output Where to save the file to
	 * @return Index of url it downloaded or -1 if it failed
	 */
	public static int download( List<String> urls , File output ) {
		for (int i = 0; i < urls.size(); i++) {
			String location = urls.get(i);
			try {
				download( location, output);
				return i;
			} catch( IOException o ) {
				System.err.println("Failed because of "+o.getClass().getSimpleName());
				System.err.println(o.getMessage());
				System.err.println();
			}
		}
		return -1;
	}

	/**
	 * Downloads the specified URL.  Throws an IOException if it fails for any reason.  Prints out
	 * information and status to console
	 *
	 * @param location Location of file
	 * @param output Where it will save the downloaded file to
	 * @throws IOException Thrown if anything goes wrong
	 */
	public static void download( String location , File output ) throws IOException {

		URL url = new URL(location);
		URLConnection connection = url.openConnection();

		connection.setConnectTimeout(1000);
		connection.connect();

		long remoteFileSize = connection.getContentLengthLong();

		System.out.println("Content length = "+remoteFileSize/1024/1024+" MB");

		if( output.exists() ) {
			if( remoteFileSize > 0 && output.length() != remoteFileSize ) {
				System.out.println("File exists, but is not the expected size.  found "+
						output.length()+" expected "+remoteFileSize);
				if( !output.delete() )
					throw new IOException("Failed to delete corrupted file");
			} else {
				System.out.println("file already downloaded");
				return;
			}
		}

		InputStream is = connection.getInputStream();
		FileOutputStream fos = new FileOutputStream(output);

		byte buffer[] = new byte[1024*100];
		long downloadedBytes = 0;

		int ticks = 0;
		int maxTicks = 60;

		System.out.println("Downloading: "+url);
		if( remoteFileSize > 0 ) {
			System.out.print("|");
			for (int i = 1; i < maxTicks; i++) {
				System.out.print("-");
			}
			System.out.println("|");
		} else {
			System.out.println("   unknown remote file size");
		}
		try {
			escape:while( downloadedBytes < remoteFileSize ) {
				while (is.available() > 0) {
					int amount = Math.min(buffer.length, is.available());
					int ret = is.read(buffer, 0, amount);
					if (ret <= 0) {
						break escape;
					}
					downloadedBytes += ret;
					fos.write(buffer, 0, ret);

					if (remoteFileSize > 0) {
						int updatedTicks = (int) (maxTicks * downloadedBytes / remoteFileSize);
						for (int i = ticks; i < updatedTicks; i++) {
							System.out.print("*");
						}
						ticks = updatedTicks;
					}
				}
			}
		} finally {
			is.close();
			fos.flush();
			fos.close();
		}
		if( remoteFileSize > 0 ) {
			int updatedTicks = (int) (maxTicks * downloadedBytes / remoteFileSize);
			for (int i = ticks; i < updatedTicks; i++) {
				System.out.print("*");
			}
			System.out.println();
		}

		if( remoteFileSize > 0 && downloadedBytes != remoteFileSize )
			throw new IOException("Didn't download the entire file.  fraction = "+(downloadedBytes/(double)remoteFileSize));
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
