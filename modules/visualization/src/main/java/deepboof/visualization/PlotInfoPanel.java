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

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import java.awt.*;
import java.util.Map;
import java.util.Set;

/**
 * @author Peter Abeles
 */
public class PlotInfoPanel extends JPanel {

	JTextPaneAA resultsText = new JTextPaneAA();
	JTextPaneAA parametersText = new JTextPaneAA();

	public PlotInfoPanel() {
		setLayout(new BoxLayout(this,BoxLayout.Y_AXIS));

		resultsText.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED));
		parametersText.setBorder(BorderFactory.createEtchedBorder(EtchedBorder.LOWERED));

		resultsText.setMaximumSize(new Dimension(Integer.MAX_VALUE,120));

//		Font font = new Font("monospaced", Font.PLAIN, 12);
//		resultsText.setFont(font);
//		parametersText.setFont(font);

		parametersText.setContentType( "text/html" );

		add(resultsText);
		add(parametersText);
	}

	public void update( GridParameterResultsApp.GridResult grid ) {
		String results = "";

		results += "    "+grid.name+"\n\n";
		results += "Testing:  "+grid.getTestScore()+"%\n";
		results += "Training: "+grid.getTrainingScore()+"%\n";
		results += "Epocs:    "+grid.test.size()+"\n";

		if( grid.testConfusion != null ) {
			ConfusionCounts confusion = grid.testConfusion;
			confusion.precompute();
			double bestScore = -Double.POSITIVE_INFINITY;
			int bestIndex = -1;
			double worstScore = Double.POSITIVE_INFINITY;
			int worstIndex = -1;

			for (int i = 0; i < confusion.N; i++) {
				double p = confusion.precision(i);
				if( p > bestScore ) {
					bestScore = p;
					bestIndex = i;
				}
				if( p < worstScore ) {
					worstScore = p;
					worstIndex = i;
				}
			}

			results += "Best:     "+confusion.labels.get(bestIndex)+"\n";
			results += String.format("          %5.1f%%\n",100.0*bestScore);
			results += "Worst:    "+confusion.labels.get(worstIndex)+"\n";
			results += String.format("          %5.1f%%\n",100.0*worstScore);
		}

		String params = "";

		Set<Map.Entry<String,String>> entries = grid.parameters.entrySet();
		for( Map.Entry<String,String> e : entries ) {
			params += "<b>"+e.getKey()+":</b><br>";
			params += "&nbsp;&nbsp;&nbsp;&nbsp;"+format(e.getValue())+"<br>";
		}

		final String fResults = results;
		final String fParams = params;

		SwingUtilities.invokeLater(()->{
			resultsText.setText(fResults);
			parametersText.setText(fParams);
		});
	}

	/**
	 * Tries to stop floating point numbers from being excessively long
	 */
	private static String format( String original ) {
		try {
			int i = Integer.parseInt(original);
			return original;
		} catch( RuntimeException ignore){}
		try {
			double v = Double.parseDouble(original);
			if( Math.abs(v) < 1e-3 || Math.abs(v) > 1e3 ) {
				return String.format("%8.2e",v);
			} else {
				return String.format("%9.5f",v);
			}
		} catch( RuntimeException ignore){}
		return original;
	}

}
