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

package deepboof.apps;

import boofcv.gui.learning.ConfusionMatrixPanel;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.graphics.Location;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.axes.Axis;
import de.erichseifert.gral.plots.axes.LinearRenderer2D;
import de.erichseifert.gral.plots.axes.LogarithmicRenderer2D;
import de.erichseifert.gral.plots.lines.SmoothLineRenderer2D;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;
import de.erichseifert.gral.plots.points.SizeablePointRenderer;
import de.erichseifert.gral.ui.InteractivePanel;
import de.erichseifert.gral.util.GraphicsUtils;
import deepboof.visualization.ConfusionCounts;
import deepboof.visualization.ConfusionFraction;
import deepboof.visualization.PlotControlPanel;
import org.ddogleg.struct.DogArray_F64;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static de.erichseifert.gral.util.GraphicsUtils.deriveDarker;
import static deepboof.visualization.GridParameterLogParser.*;

/**
 * @author Peter Abeles
 */
// TODO Click history chart and show performance at that point
// TODO Select from list
//    Sort list by name, score (test and training), and number of epocs
// TODO Data Sets Summary Table
//    total in test and training
//    total for each category
//    F-statistic, prevision, recall: total and for each category
// TODO reload option
// TODO Confusion plot
//    Clarify what rows and columns are in the matrix
// TODO Confusion vs Epoc
//    Play video with time line
//    Select and plot vs epoc individual elements in confusion

public class GridParameterResultsApp implements PlotControlPanel.Listener {

	@Parameter(names = { "-h", "--Help" }, description = "Prints argument help")
	public boolean help = false;

	@Parameter(names = { "-p", "--InputPath" }, description = "Path to where results are contained")
	public String inputPath = "./";

	@Parameter(names = { "-r", "--ResultsName" }, description = "All results directories contain this word")
	public String resultsName = "results";

	@Parameter(names = { "-n", "--ParameterName" }, description = "Name of parameter file")
	public String parameterName = "parameter.txt";

	@Parameter(names = { "-tr", "--TrainingName" }, description = "Name of log file with training set results")
	public String trainingName = "train.txt";

	@Parameter(names = { "-te", "--TestName" }, description = "Name of log file with test set results")
	public String testName = "test.txt";

	@Parameter(names = { "-cr", "--TrainingConfusion" },
			description = "Name of log file with confusion matrix for training")
	public String confusionTrainingName = "test_confusion.txt";

	@Parameter(names = { "-ce", "--TestConfusion" }, description = "Name of log file with confusion matrix for test")
	public String confusionTestName = "test_confusion.txt";

	boolean percent = true;

	List<GridResult> results = new ArrayList<>();
	List<String> parameters = new ArrayList<>();

	int selectedIndex = 0;

	String param0,param1;

	private double maxClickDistance = 15.0;

	DataTable dataSpecific = new DataTable(Double.class, Double.class, Double.class);
	DataTable dataOverview = new DataTable(Double.class, Double.class, Double.class);

	XYPlot specificPlot;
	InteractivePanel specificPlotPanel;

	XYPlot overviewPlot;
	CustomInteractivePanel overviewPlotPanel;

	PlotControlPanel control = new PlotControlPanel(this);
	PlotInfoPanel infoPanel = new PlotInfoPanel();

	boolean hasConfusion = true;
	ConfusionMatrixPanel confusionTrainPanel = new ConfusionMatrixPanel(600,true);
	ConfusionMatrixPanel confusionTestPanel = new ConfusionMatrixPanel(600,true);

	boolean guiInitialized = false;

	public GridParameterResultsApp() {
		confusionTrainPanel.setShowLabels(true);
		confusionTrainPanel.setShowZeros(false);
		confusionTrainPanel.setGray(true);
		confusionTrainPanel.addMouseListener(new ConfusionMouse(confusionTrainPanel));

		confusionTestPanel.setShowLabels(true);
		confusionTestPanel.setShowZeros(false);
		confusionTestPanel.setGray(true);
		confusionTestPanel.addMouseListener(new ConfusionMouse(confusionTestPanel));
	}

	public void run() {
		if( !new File(inputPath).exists() ) {
			System.err.println("The requested input path does not exist.  '"+inputPath+"'");
			System.exit(1);
		}

		for(File f  : new File(inputPath).listFiles() ) {
			if( !f.isDirectory() || !f.getName().contains(resultsName) )
				continue;

			GridResult r = new GridResult();
			r.name = f.getName();
			try {
				parsePerformanceEpoc(new File(f,trainingName),r.train);
				parsePerformanceEpoc(new File(f,testName),r.test);
				parseParameters(new File(f,parameterName),r.parameters);

				// parse optional files now
				r.trainConfusion = parseConfusion(new File(f,confusionTrainingName));
				r.testConfusion = parseConfusion(new File(f,confusionTestName));

				hasConfusion &= r.trainConfusion != null && r.testConfusion != null;

				results.add(r);
			} catch( IOException e ) {
				System.out.println("Skipping "+f.getName()+" because "+e.getMessage());
			} catch( RuntimeException e ) {
				System.out.println("Skipping "+f.getName()+" because "+e.getMessage());
				e.printStackTrace();
			}
		}
		if( results.isEmpty() ) {
			System.err.println("No data results found in "+inputPath);
			System.exit(1);
		}

		extractNumericalParameters();

		System.out.println("Total read: "+results.size());
		System.out.println("Total parameters "+parameters.size());

		control.setParameters(parameters);
		infoPanel.update(results.get(0));

		JTabbedPane tabbedPane = new JTabbedPane();
		tabbedPane.addTab("History", createSpecificPlot());
		if( hasConfusion ) {
			tabbedPane.addTab("Confusion Test", confusionTestPanel);
			tabbedPane.addTab("Confusion Train", confusionTrainPanel);
		}
		tabbedPane.setPreferredSize(tabbedPane.getComponentAt(0).getPreferredSize());

		JPanel overviewPanel = new JPanel();
		overviewPanel.setLayout(new BoxLayout(overviewPanel,BoxLayout.Y_AXIS));
		overviewPanel.add(createOverviewPlot());
		overviewPanel.add(control);

		JSplitPane plots = new JSplitPane(JSplitPane.VERTICAL_SPLIT, overviewPanel, tabbedPane);
		plots.setResizeWeight(0.5);
		plots.setOneTouchExpandable(true);
		plots.setContinuousLayout(true);

		guiInitialized = true;
		JPanel gui = new JPanel();
		gui.setLayout(new BorderLayout());
		gui.add(plots,BorderLayout.CENTER);
		gui.add(infoPanel,BorderLayout.EAST);

		focusOnResult(0);

		final JFrame frame = new JFrame("Training Grid Results");
		frame.add(gui, BorderLayout.CENTER);
		frame.setJMenuBar(createMenuBar());
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setPreferredSize(new Dimension(800,800));
		frame.pack();
		frame.setVisible(true);
	}

	private JMenuBar createMenuBar() {
		JMenuBar menuBar = new JMenuBar();
		JMenu selectMenu = new JMenu("Select");
		menuBar.add(selectMenu);

		JMenuItem menuItem = new JMenuItem("Best Test");
		selectMenu.add(menuItem);
		menuItem.addActionListener((ActionEvent e)->{selectBestTest();});

		menuItem = new JMenuItem("Best Training");
		selectMenu.add(menuItem);
		menuItem.addActionListener((ActionEvent e)->{selectBestTraining();});

		menuItem = new JMenuItem("Most Epocs");
		selectMenu.add(menuItem);
		menuItem.addActionListener((ActionEvent e)->{selectMostEpocs();});

		menuItem = new JMenuItem("From List");
		selectMenu.add(menuItem);


		JMenu windowMenu = new JMenu("Window");
		menuBar.add(windowMenu);
		menuItem = new JMenuItem("Show Confusion");
		windowMenu.add(menuItem);

		return menuBar;
	}

	private void selectBestTest() {
		double best = -Double.MAX_VALUE;
		int bestIndex = -1;
		for (int i = 0; i < results.size(); i++) {
			GridResult r = results.get(i);
			double s = r.getTestScore();
			if( s > best ) {
				best = s;
				bestIndex = i;
			}
		}
		focusOnResult(bestIndex);
	}

	private void selectBestTraining() {
		double best = -Double.MAX_VALUE;
		int bestIndex = -1;
		for (int i = 0; i < results.size(); i++) {
			GridResult r = results.get(i);
			double s = r.getTrainingScore();
			if( s > best ) {
				best = s;
				bestIndex = i;
			}
		}
		focusOnResult(bestIndex);
	}

	private void selectMostEpocs() {
		int best = Integer.MIN_VALUE;
		int bestIndex = -1;
		for (int i = 0; i < results.size(); i++) {
			GridResult r = results.get(i);
			int s = r.test.size;
			if( s > best ) {
				best = s;
				bestIndex = i;
			}
		}
		focusOnResult(bestIndex);
	}



	private void extractNumericalParameters() {
		List<String> parametersAll = new ArrayList<>();
		for( GridResult r : results ) {
			for( String s : r.parameters.keySet() ) {
				if( !parametersAll.contains(s) ) {
					parametersAll.add(s);
				}
			}
		}
		// make sure it's a numerical parameter
		GridResult r = results.get(0);
		for( String s : parametersAll ) {
			try {
				Double.parseDouble(r.parameters.get(s));
				parameters.add(s);
			} catch( RuntimeException ignore){}
		}
	}

	private JPanel createSpecificPlot() {

		GridResult grid = results.get(selectedIndex);

		for (int i = 0; i < grid.test.size(); i++) {
			dataSpecific.add((double)i, grid.test.get(i), grid.train.get(i) );
		}

		DataSeries seriesTest  = new DataSeries("Test",dataSpecific, 0, 1);
		DataSeries seriesTrain = new DataSeries("Training",dataSpecific, 0, 2);

		// Create a new xy-plot
		final XYPlot plot = new XYPlot(seriesTest,seriesTrain);

		// Format plot
		plot.setInsets(new Insets2D.Double(60, 60, 60, 20));
		plot.getTitle().setText(grid.name);
		plot.setLegendVisible(true);
		plot.setLegendLocation(Location.NORTH);

		specificPlot = plot;

		SmoothLineRenderer2D renderTest = new SmoothLineRenderer2D();
		renderTest.setColor(deriveDarker(Color.RED));
		PointRenderer renderTestPoint = new DefaultPointRenderer2D();
		renderTestPoint.setColor(renderTest.getColor());
		plot.setLineRenderers(seriesTest, renderTest);
		plot.setPointRenderers(seriesTest, renderTestPoint);

		SmoothLineRenderer2D renderTrain = new SmoothLineRenderer2D();
		renderTrain.setColor(deriveDarker(Color.BLUE));
		PointRenderer renderTrainPoint = new DefaultPointRenderer2D();
		renderTrainPoint.setColor(renderTrain.getColor());
		plot.setLineRenderers(seriesTrain, renderTrain);
		plot.setPointRenderers(seriesTrain, renderTrainPoint);

		plot.getAxisRenderer(XYPlot.AXIS_X).getLabel().setText("Epoc");
		plot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setText("Score");
		plot.getAxis(XYPlot.AXIS_Y).setAutoscaled(false);
		plot.getAxis(XYPlot.AXIS_Y).setRange(0,101);


		specificPlotPanel = new InteractivePanel(plot);
		JPanel wrapper = new JPanel();
		wrapper.setLayout(new BorderLayout());
		wrapper.add(specificPlotPanel,BorderLayout.CENTER);
		wrapper.setBackground(Color.WHITE);
		return wrapper;
	}

	private InteractivePanel createOverviewPlot() {

		double scale = percent ? 0.2 : 20.0;

		// Create a new xy-plot
		DataSeries series = new DataSeries(dataOverview);
		updateOverviewData();
		final XYPlot plot = new XYPlot(series);

		// Format plot
		plot.setInsets(new Insets2D.Double(20, 80, 60, 20));
		plot.getTitle().setText(param0+" Vs "+param1);

		overviewPlot = plot;

		CustomInteractivePanel panel = new CustomInteractivePanel(plot,this);
		overviewPlotPanel = panel;

		// Format points
		Color color = GraphicsUtils.deriveWithAlpha(Color.RED, 96);
		SizeablePointRenderer pointRenderer = new SizeablePointRenderer();
		pointRenderer.setShape(new Ellipse2D.Double(-0.5*scale, -0.5*scale, scale, scale));  // shape of data points
		pointRenderer.setColor(color);  // color of data points
		pointRenderer.setColumn(2);  // data column which determines the scaling of data point shapes
		plot.setPointRenderers(series, pointRenderer);  // Assign the point renderer to the data series

		configureScatterAxis(control.scatterLog);

		panel.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				Axis axisX = plot.getAxis(XYPlot.AXIS_X);
				Axis axisY = plot.getAxis(XYPlot.AXIS_Y);

				double bestDistance = Double.MAX_VALUE;
				int bestIndex = -1;

				double offX = plot.getPlotArea().getX();
				double offY = plot.getPlotArea().getY();

				double height = plot.getPlotArea().getHeight();

				for (int i = 0; i < results.size(); i++) {
					GridResult r = results.get(i);
					double v0 = Double.parseDouble(r.parameters.get(param0));
					double v1 = Double.parseDouble(r.parameters.get(param1));

					double pixelX = plot.getAxisRenderer(XYPlot.AXIS_X).worldToView(axisX,v0,true);
					double pixelY = plot.getAxisRenderer(XYPlot.AXIS_Y).worldToView(axisY,v1,true);

					pixelX = offX + pixelX;
					pixelY = offY + height - pixelY;

					double dx = pixelX-e.getX();
					double dy = pixelY-e.getY();

					double d = dx*dx + dy*dy;
					if( d < bestDistance ) {
						bestDistance = d;
						bestIndex = i;
					}
				}

				if( bestDistance < maxClickDistance*maxClickDistance && bestIndex != -1 ) {
					focusOnResult(bestIndex);
				}
			}
		});

		return panel;
	}

	private void focusOnResult(int index) {
		selectedIndex = index;

		GridResult grid = results.get(selectedIndex);
		dataSpecific.clear();
		for (int i = 0; i < grid.test.size(); i++) {
			dataSpecific.add((double)i, grid.test.get(i), grid.train.get(i) );
		}
		infoPanel.update(grid);
		specificPlot.getTitle().setText(grid.name);
		specificPlotPanel.repaint();
		overviewPlotPanel.repaint(); // update what's selected

		if( hasConfusion ) {
			confusionTestPanel.setLabels(grid.testConfusion.labels);
			ConfusionFraction cf = new ConfusionFraction(grid.testConfusion);
			confusionTestPanel.setMatrix(cf.M);
			confusionTestPanel.repaint();

			confusionTrainPanel.setLabels(grid.trainConfusion.labels);
			cf = new ConfusionFraction(grid.trainConfusion);
			confusionTrainPanel.setMatrix(cf.M);
			confusionTrainPanel.repaint();
		}
	}

	private void updateOverviewData() {
		param0 = parameters.get(control.scatterX);
		param1 = parameters.get(control.scatterY);

		dataOverview.clear();
		for (int i = 0; i < results.size(); i++) {
			GridResult r = results.get(i);
			double v0 = Double.parseDouble(r.parameters.get(param0));
			double v1 = Double.parseDouble(r.parameters.get(param1));

			dataOverview.add(v0, v1, r.getTestScore());
		}
	}

	@Override
	public void uiScatterLog(boolean showLog) {
		configureScatterAxis(showLog);

		overviewPlotPanel.repaint();
	}

	private void configureScatterAxis(boolean showLog) {
		sanityCheckGuiThread();

		XYPlot plot = overviewPlot;
		if( showLog ) {
			plot.setAxisRenderer(XYPlot.AXIS_X,new LogarithmicRenderer2D());
			plot.setAxisRenderer(XYPlot.AXIS_Y,new LogarithmicRenderer2D());
		} else {
			plot.setAxisRenderer(XYPlot.AXIS_X,new LinearRenderer2D());
			plot.setAxisRenderer(XYPlot.AXIS_Y,new LinearRenderer2D());
		}

		plot.getAxisRenderer(XYPlot.AXIS_X).setTickLabelFormat(new DecimalFormat("0.##E0"));
		plot.getAxisRenderer(XYPlot.AXIS_Y).setTickLabelFormat(new DecimalFormat("0.##E0"));
		plot.getAxisRenderer(XYPlot.AXIS_X).setTickLabelRotation(60);
		plot.getAxisRenderer(XYPlot.AXIS_Y).setTickLabelRotation(60);
		plot.getAxisRenderer(XYPlot.AXIS_X).setTickAlignment(1.25);
		plot.getAxisRenderer(XYPlot.AXIS_Y).setTickAlignment(1.5);

		plot.getAxisRenderer(XYPlot.AXIS_X).getLabel().setText(param0);
		plot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setText(param1);
		plot.getAxisRenderer(XYPlot.AXIS_X).setLabelDistance(2);
		plot.getAxisRenderer(XYPlot.AXIS_Y).setLabelDistance(2);
	}

	private void sanityCheckGuiThread() {
		if( guiInitialized && !SwingUtilities.isEventDispatchThread() )
			throw new RuntimeException("Not in GUI thread!");
	}

	@Override
	public void uiScatterSelected(int labelX, int labelY) {
		updateOverviewData();
		overviewPlot.getTitle().setText(param0+" vs "+param1);
		overviewPlot.getAxisRenderer(XYPlot.AXIS_X).getLabel().setText(param0);
		overviewPlot.getAxisRenderer(XYPlot.AXIS_Y).getLabel().setText(param1);
		overviewPlotPanel.repaint();
	}

	/**
	 * Mouse listener which turns on and off highlighting when a category is clicked
	 */
	private static class ConfusionMouse extends MouseAdapter {
		ConfusionMatrixPanel owner;
		ConfusionMatrixPanel.LocationInfo info;
		int previous = -1;

		public ConfusionMouse(ConfusionMatrixPanel owner) {
			this.owner = owner;
		}

		@Override
		public void mousePressed(MouseEvent e) {
			info = owner.whatIsAtPoint(e.getX(),e.getY(),info);

			if( info.insideMatrix ) {

			} else {
				if( previous == info.col ) {
					owner.setHighlightCategory(-1);
				} else {
					owner.setHighlightCategory(info.col);
				}
				previous = owner.getHighlightCategory();
				owner.repaint();
			}
		}
	}

	public static class GridResult {
		String name = "UNSPECIFIED";
		DogArray_F64 train = new DogArray_F64();
		DogArray_F64 test = new DogArray_F64();

		ConfusionCounts trainConfusion;
		ConfusionCounts testConfusion;

		Map<String,String> parameters = new HashMap<>();

		public double getTrainingScore() {
			return train.get( train.size-1);
		}
		public double getTestScore() {
			return test.get( train.size-1);
		}
	}

	// java -jar modules/visualization/ResultsPlot.jar -p results -n training_parameters.txt -r grid -te test.log -tr train.log
	public static void main(String[] args) {
		GridParameterResultsApp app = new GridParameterResultsApp();
		JCommander jCommander;
		try {
			jCommander = new JCommander(app, args);
		} catch( ParameterException e ) {
			System.out.println(e.getMessage());
			System.out.println();
			System.out.println("Use --Help to print out help");
			return;
		}
		jCommander.setProgramName("Results Plotting");
		if (app.help) {
			jCommander.usage();
			return;
		}
		app.run();
	}
}
