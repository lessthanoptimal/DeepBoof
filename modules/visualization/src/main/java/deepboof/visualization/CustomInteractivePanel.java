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

import de.erichseifert.gral.graphics.Drawable;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.axes.Axis;
import de.erichseifert.gral.ui.InteractivePanel;
import georegression.struct.point.Point2D_F64;

import java.awt.*;
import java.util.List;

/**
 * Custom panel which will draw the currently selected object on top of the plot
 *
 * @author Peter Abeles
 */
public class CustomInteractivePanel extends InteractivePanel {

	GridParameterResultsApp app;

	public CustomInteractivePanel(Drawable drawable, GridParameterResultsApp app) {
		super(drawable);
		this.app = app;
	}


	@Override
	public void paintComponent( Graphics g ) {
		super.paintComponent(g);

		Graphics2D g2 = (Graphics2D)g;
		List<GridParameterResultsApp.GridResult> results = app.results;
		int selectedIndex = app.selectedIndex;
		if( selectedIndex < 0 || selectedIndex >= results.size() ) {
			return;
		}

		int circleRadius = 10;
		int circleWidth = 2*circleRadius + 1;

		GridParameterResultsApp.GridResult grid = results.get(selectedIndex);

		double value0 = Double.parseDouble(grid.parameters.get( app.param0 ));
		double value1 = Double.parseDouble(grid.parameters.get( app.param1 ));

		Point2D_F64 pixel = plotToPixel(value0,value1,app.overviewPlot);

		pixel.x -= circleRadius;
		pixel.y -= circleRadius;

		g2.setColor(Color.BLACK);
		g2.drawOval((int)(pixel.x+0.5),(int)(pixel.y+0.5),circleWidth,circleWidth);
	}

	public static Point2D_F64 plotToPixel( double x , double y , XYPlot plot ) {
		Axis axisX = plot.getAxis(XYPlot.AXIS_X);
		Axis axisY = plot.getAxis(XYPlot.AXIS_Y);
		double offX = plot.getPlotArea().getX();
		double offY = plot.getPlotArea().getY();
		double height = plot.getPlotArea().getHeight();

		double pixelX = plot.getAxisRenderer(XYPlot.AXIS_X).worldToView(axisX,x,true);
		double pixelY = plot.getAxisRenderer(XYPlot.AXIS_Y).worldToView(axisY,y,true);

		Point2D_F64 out = new Point2D_F64();
		out.x =  offX + pixelX;
		out.y = offY + height - pixelY - 1;

		return out;
	}
}
