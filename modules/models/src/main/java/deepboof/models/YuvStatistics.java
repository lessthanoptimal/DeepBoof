package deepboof.models;

/**
 * @author Peter Abeles
 */
public class YuvStatistics {
	public double meanU,meanV;
	public double stdevU,stdevV;
	public double kernel[];
	public int kernelOffset; // index offset from zero for the kernel. i.e. it's center
	public String border;
}
