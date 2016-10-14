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

/**
 * @author Peter Abeles
 */
public class DeepVisulization {

	public static JFrame showWindow( final JComponent component , String title ) {
		return showWindow(component,title,false);
	}

	public static JFrame showWindow( final JComponent component , String title, final boolean closeOnExit ) {
		final JFrame frame = new JFrame(title);
		frame.add(component, BorderLayout.CENTER);

		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				frame.pack();
				frame.setVisible(true);
				if( closeOnExit )
					frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			}
		});

		return frame;
	}
}
