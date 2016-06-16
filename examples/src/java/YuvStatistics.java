import boofcv.struct.convolve.Kernel1D_F32;

/**
 * @author Peter Abeles
 */
public class YuvStatistics {
	public double meanU,meanV;
	public double stdevU,stdevV;
	public double kernel[];
	public String border;

	public Kernel1D_F32 create1D_F32() {
		Kernel1D_F32 k = new Kernel1D_F32(kernel.length);
		for (int i = 0; i < kernel.length; i++) {
			k.data[i] = (float)kernel[i];
		}
		return k;
	}
}
