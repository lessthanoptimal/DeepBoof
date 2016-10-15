package deepboof.models;


import java.io.*;

/**
 * @author Peter Abeles
 */
public class DeepModelIO {

	public static void save(YuvStatistics params , File file ) throws FileNotFoundException {
		PrintStream out = new PrintStream(file);

		out.printf("meanU %f\n",params.meanU);
		out.printf("meanV %f\n",params.meanV);
		out.printf("stdevU %f\n",params.stdevU);
		out.printf("stdevV %f\n",params.stdevV);
		out.printf("border %s\n",params.border);
		out.printf("kernelOffset %d\n",params.kernelOffset);
		out.print("kernel");
		for (int i = 0; i < params.kernel.length; i++) {
			out.printf(" %.10f",params.kernel[i]);
		}
		out.println();
		out.close();
	}

	public static YuvStatistics load(File file ) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));

		YuvStatistics out = new YuvStatistics();
		out.meanU = readDouble(reader.readLine());
		out.meanV = readDouble(reader.readLine());
		out.stdevU = readDouble(reader.readLine());
		out.stdevV = readDouble(reader.readLine());
		out.border = readString(reader.readLine());
		out.kernelOffset = readInt(reader.readLine());
		out.kernel = readArray(reader.readLine());

		return out;
	}

	private static String readString( String line ) {
		return line.split(" ")[1];
	}

	private static int readInt( String line ) {
		return Integer.parseInt(line.split(" ")[1]);
	}

	private static double readDouble( String line ) {
		return Double.parseDouble(line.split(" ")[1]);
	}

	private static double[] readArray( String line ) {
		String words[] = line.split(" ");
		double[] out = new double[ words.length-1 ];
		for (int i = 0; i < out.length; i++) {
			out[i] = Double.parseDouble(words[i+1]);
		}
		return out;
	}

}
