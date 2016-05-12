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

package deepboof.visualization;

import org.ddogleg.struct.GrowQueue_F64;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

/**
 * @author Peter Abeles
 */
public class GridParameterLogParser {

	/**
	 * Reads a file where each line contains the accuracy at the end of each epoc during training
	 * @param file File which is to be parsed
	 * @param storage Where the results are to be stored
	 * @throws IOException thrown if an error occurs
	 */
	public static void parsePerformanceEpoc( File file , GrowQueue_F64 storage ) throws IOException
	{
		if( !file.exists() ) throw new IOException(file.getName()+" doesn't exist");

		BufferedReader reader = new BufferedReader(new FileReader(file));

		storage.reset();

		String line;
		while ((line = reader.readLine()) != null) {
			if( line.charAt(0) == '%' || line.charAt(0) == '#')
				continue;

			storage.add( Double.parseDouble(line.trim()));
		}
	}

	public static void parseParameters( File file , Map<String,String> storage ) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(file));

		String line;
		while ((line = reader.readLine()) != null) {
			if( line.charAt(0) == '%' || line.charAt(0) == '#')
				continue;

			String words[] = line.trim().split(" ");
			if( words.length != 2 )
				throw new IOException("Expected two words for each line: "+line.trim());

			storage.put(words[0],words[1]);
		}
	}

	public static ConfusionCounts parseConfusion( File file ) throws IOException {
		if( !file.exists() )
			return null;

		BufferedReader reader = new BufferedReader(new FileReader(file));

		String words[] = reader.readLine().split(" ");

		ConfusionCounts out = new ConfusionCounts(Arrays.asList(words));

		int N = out.N;
		for (int row = 0; row < N; row++) {
			String line = reader.readLine();
			if( line == null)
				throw new IOException("Premature ending at line "+row);
			words = line.split(" ");
			if( words.length != N )
				throw new IOException("Expected "+N+" words on line "+row+" got "+words.length+" instead");

			for (int col = 0; col < N; col++) {
				int value = Integer.parseInt(words[col]);
				out.set(row,col, value);
			}
		}

		return out;
	}
}
