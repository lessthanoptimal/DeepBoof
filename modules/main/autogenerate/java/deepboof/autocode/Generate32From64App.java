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

package deepboof.autocode;

import com.peterabeles.auto64fto32f.ConvertFile32From64;
import com.peterabeles.auto64fto32f.RecursiveConvert;

import java.io.File;


/**
 * Auto generates 32bit code from 64bit code.
 *
 * @author Peter Abeles
 */
public class Generate32From64App extends RecursiveConvert {


	public Generate32From64App(ConvertFile32From64 converter) {
		super(converter);
	}

	public static void main(String args[] ) {
		String directories[] = new String[]{
				"modules/main/src/main/java",
				"modules/main/src/test/java",
				"modules/io/src/main/java",};

		ConvertFile32From64 converter = new ConvertFile32From64(false);

		converter.replacePattern( "/\\*\\*/getDouble", "__GET_DOUBLE" );
		converter.replacePattern("/\\*\\*/double", "FIXED_DOUBLE");
		converter.replacePattern( "DOUBLE_TEST_TOL", "FLOAT_TEST_TOL");
		converter.replacePattern( "TEST_TOL_F64", "TEST_TOL_F32" );
		converter.replacePattern( "TEST_TOL_A_F64", "TEST_TOL_A_F32" );
		converter.replacePattern( "TEST_TOL_B_F64", "TEST_TOL_B_F32" );
		converter.replacePattern("double", "float");
		converter.replacePattern("Double", "Float");
		converter.replacePattern("_F64", "_F32");

		converter.replaceStartsWith("Math.", "(float)Math.");
		converter.replaceStartsWith("-Math.", "(float)-Math.");
		converter.replaceStartsWith( "rand.nextGaussian", "(float)rand.nextGaussian" );

		converter.replacePatternAfter("FIXED_DOUBLE", "/\\*\\*/double");
		converter.replacePatternAfter( "__GET_DOUBLE","/\\*\\*/getDouble" );

		Generate32From64App app = new Generate32From64App(converter);
		for( String dir : directories ) {
			app.process(new File(dir) );
		}
	}
}
