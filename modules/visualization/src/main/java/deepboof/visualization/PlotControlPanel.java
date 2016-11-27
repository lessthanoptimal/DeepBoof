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
import java.awt.*;
import java.awt.event.*;
import java.util.List;

/**
 * Displays controls for adjusting what is shown in plots and how it is shown as well as displaying
 * summary statistics.
 *
 * @author Peter Abeles
 */
public class PlotControlPanel extends JPanel
		implements ItemListener, ActionListener
{

	JCheckBox cScatterLog;
	JComboBox<String> cSelectX;
	JComboBox<String> cSelectY;

	// which thing parameters are shown in scatter plot
	public int scatterX=0,scatterY=1;
	// show the scatter plot with a log scale
	public boolean scatterLog=false;

	Listener listener;

	public PlotControlPanel(Listener listener ) {
		this.listener = listener;
		setLayout(new BoxLayout(this,BoxLayout.X_AXIS));

		cScatterLog = new JCheckBox("Log Plot");
		cScatterLog.setMnemonic(KeyEvent.VK_L);
		cScatterLog.setSelected(scatterLog);
		cScatterLog.addItemListener(this);

		cSelectX = new JComboBox<>();
		cSelectY = new JComboBox<>();

		add(cScatterLog);
		add(Box.createHorizontalGlue());
		add(new JLabel("X: "));
		add(cSelectX);
		add(Box.createRigidArea(new Dimension(5,0)));
		add(new JLabel("Y: "));
		add(cSelectY);
	}

	public void setParameters( final List<String> parameters ) {
		if( parameters.size() < 1 )
			throw new IllegalArgumentException("There needs to be at least one parameter");
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				DefaultComboBoxModel<String> modelX = (DefaultComboBoxModel<String>)cSelectX.getModel();
				modelX.removeAllElements();
				for( String p : parameters ) {
					modelX.addElement(p);
				}

				modelX = (DefaultComboBoxModel<String>)cSelectY.getModel();
				modelX.removeAllElements();
				for( String p : parameters ) {
					modelX.addElement(p);
				}

				scatterX = 0;
				scatterY = Math.min(parameters.size(),1);

				cSelectX.invalidate();
				cSelectY.invalidate();

				cSelectX.setPreferredSize(cSelectX.getMinimumSize());
				cSelectX.setMaximumSize(cSelectX.getMinimumSize());
				cSelectY.setPreferredSize(cSelectY.getMinimumSize());
				cSelectY.setMaximumSize(cSelectY.getMinimumSize());

				// prevent it from sending out an event for a change in selected index
				cSelectX.removeActionListener(PlotControlPanel.this);
				cSelectY.removeActionListener(PlotControlPanel.this);

				cSelectX.setSelectedIndex(scatterX);
				cSelectY.setSelectedIndex(scatterY);

				cSelectX.addActionListener(PlotControlPanel.this);
				cSelectY.addActionListener(PlotControlPanel.this);
			}});
		}

		@Override
		public void itemStateChanged(ItemEvent e) {
			if( e.getSource() == cScatterLog ) {
				scatterLog = cScatterLog.isSelected();
				listener.uiScatterLog(scatterLog);
			}
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			if( e.getSource() == cSelectX ) {
			scatterX = cSelectX.getSelectedIndex();
			listener.uiScatterSelected(scatterX,scatterY);
		} else if( e.getSource() == cSelectY ) {
			scatterY = cSelectY.getSelectedIndex();
			listener.uiScatterSelected(scatterX,scatterY);
		}
	}

	public interface Listener {
		void uiScatterLog( boolean showLog );

		void uiScatterSelected( int labelX , int labelY );
	}
}
