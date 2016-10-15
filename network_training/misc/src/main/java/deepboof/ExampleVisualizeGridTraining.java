package deepboof;

import deepboof.apps.GridParameterResultsApp;

/**
 * Example of how to visualize logged data generated during a grid search of optimization parameter
 * space.  You will first need to run the example found in DeepBoof/examples/src/torch.  Then you can
 * run this class to visualize the results, or invoke the command line arguments below.  Once you
 * jave built the jar only the last line is needed.
 *
 * <pre>
 * cd DeepBoof
 * gradle resultsPlotJar
 * mv modules/visualization/ResultsPlot.jar  .
 * java -jar ResultsPlot.jar -p examples/src/torch/results/ --ParameterName training_parameters.txt
 *      --TestConfusion confusion_test.txt --TestName test.log
 *      --TrainingConfusion confusion_train.txt --TrainingName train.log --ResultsName grid
 *  </pre>
 * @author Peter Abeles
 */
public class ExampleVisualizeGridTraining {
	public static void main(String[] args) {
		GridParameterResultsApp app = new GridParameterResultsApp();
		app.inputPath = "examples/src/torch/results/";
		app.parameterName = "training_parameters.txt";
		app.confusionTestName = "confusion_test.txt";
		app.confusionTrainingName = "confusion_train.txt";
		app.resultsName = "grid";
		app.trainingName = "train.log";
		app.testName = "test.log";

		app.run();
	}
}
