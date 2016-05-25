package deepboof.io.torch7;

import deepboof.io.torch7.struct.TorchObject;
import deepboof.misc.TensorOps;
import org.junit.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author Peter Abeles
 */
public class TestSerializeBinaryTorch7 {
	File pathToInput = new File(TensorOps.pathToRoot(),"modules/io/src/test/torch7/torch_data");
	File pathToOutput = new File(TensorOps.pathToRoot(),"modules/io/src/test/torch7/boof_data");

	/**
	 * Reads in torch data generated in torch and writes it back out as torch data
	 */
	@Test
	public void readTorchThenSaveAsTorch() throws IOException {

		pathToOutput.mkdirs();

		ParseBinaryTorch7 parseBinary = new ParseBinaryTorch7();
		SerializeBinaryTorch7 serializeBinary = new SerializeBinaryTorch7(true);

		for( File f : pathToInput.listFiles() ) {
			List<TorchObject> list = parseBinary.parse(f);

			assertEquals(1,list.size());

			OutputStream out = new FileOutputStream(new File(pathToOutput,f.getName()));
			serializeBinary.serialize(list,out);
			out.close();
		}
	}
}