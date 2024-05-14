from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import sys

import numpy as np
from time import perf_counter

from energy_harvest import EnergyHarvester
from data_utils import *

import pickle

# ====================== global settings ======================
pg.setConfigOptions(antialias=True, background="w", foreground="k")


# ====================== placeholder class ======================
class Color(QWidget):

	def __init__(self, color):
		super(Color, self).__init__()
		self.setAutoFillBackground(True)

		palette = self.palette()
		palette.setColor(QPalette.Window, QColor(color))
		self.setPalette(palette)


# ====================== maingui class ======================
class IoTDIDemo(QMainWindow):
	def __init__(self, body_parts, title, label_map, label_stream, data_stream, time_ax, data_packets, e_plots,thresh):
		super(IoTDIDemo, self).__init__()
		
		self.title = title
		self.body_parts = body_parts
		self.label_map = label_map
		self.label_stream = label_stream
		self.data_stream = data_stream
		self.time_ax = time_ax
		self.data_packets = data_packets
		self.e_plots = e_plots
		self.thresh = thresh

		self.initUI()

	def initUI(self):
		# initial GUI params
		xpos = 0
		ypos = 0
		width = 640*2
		height = int(480*1.5)
		self.plot_window_width = 20
		self.fs = 25

		self.setGeometry(xpos, ypos, width, height)
		self.setWindowTitle(self.title)

		# layout
		self.main_horizontal_pane = QHBoxLayout()
		self.left_vertical_pane = QVBoxLayout()
		self.right_vertical_pane = QVBoxLayout()
		self.checkbox_pane = QVBoxLayout()
		self.left_empty_pane = QVBoxLayout()
		self.label_scroll_pane = QVBoxLayout()
		self.plot_pane = QVBoxLayout()

		# checkbox pane
		self.checkbox_label = QLabel("Body Parts")
		self.checkbox_label.setAlignment(Qt.AlignCenter)
		font = QFont()
		font.setBold(True)  # Make the font bold
		font.setPointSize(16)  # Set the font size to 16 points
		self.checkbox_label.setFont(font)
		self.checkbox_pane.addWidget(self.checkbox_label)

		self.checkboxes = {bp: None for bp in self.body_parts}
		colors = ['red','orange','green','blue','purple']
		# self.plot_widgets = {bp: Color(colors[bp_i]) for bp_i, bp in enumerate(self.body_parts)}
		self.plot_widgets = {bp: None for bp_i, bp in enumerate(self.body_parts)}
		self.checked = []
		for bp_i, bp in enumerate(self.body_parts):
			box = QCheckBox(bp)
			box.setFont(font)
			box.stateChanged.connect(self.update_plot_layout)
			self.checkbox_pane.addWidget(box)
			self.checkbox_pane.addStretch(1)
			self.checkboxes[bp] = box
			

		# left side of GUI
		# self.left_empty_pane.addWidget(Color('red'))
		self.timer = pg.QtCore.QTimer()
		self.sbutton = QtWidgets.QPushButton("Start / Continue")
		self.ebutton = QtWidgets.QPushButton("Stop")
		self.sbutton.clicked.connect(self.start)
		self.ebutton.clicked.connect(self.stop)
		self.ebutton.setEnabled(False)
		self.timer.timeout.connect(self.time_update)
		self.left_empty_pane.addWidget(self.sbutton)
		self.left_empty_pane.addWidget(self.ebutton)
		self.start_time = 0
		self.left_vertical_pane.addLayout(self.checkbox_pane)
		self.left_vertical_pane.addLayout(self.left_empty_pane)
		self.left_empty_pane.addStretch(1)
		self.left_vertical_pane.setStretch(0,1)
		self.left_vertical_pane.setStretch(1,1)

		self.main_horizontal_pane.addLayout(self.left_vertical_pane)

		# plotting pane
		self.label_stream_widget = pg.PlotWidget()
		self.label_stream_widget.setLabel('left', "Label")
		self.my_font = QFont("Times", 12, QFont.Bold)
		self.label_stream_widget.getAxis("left").label.setFont(self.my_font)
		self.label_stream_widget.showGrid(x = True, y = True)
		self.label_stream_widget_curve = self.label_stream_widget.plot(self.time_ax[0:self.plot_window_width*self.fs],self.label_stream[0:self.plot_window_width*self.fs], pen={"color": "#2196F3", "width": 1})
		self.label_stream_widget.setYRange(0, 18)
		self.label_stream_widget.getAxis("left").setWidth(56.5)
		range_ = self.label_stream_widget.getViewBox().viewRange() 
		self.label_stream_widget.getViewBox().setLimits(yMin=range_[1][0], yMax=range_[1][1], minYRange = range_[1][1]-range_[1][0])  


		self.transitions = [i for i in range(1, len(self.label_stream)) if self.label_stream[i] != self.label_stream[i - 1]]
		self.transitions.insert(0,0)
		self.transitions = np.array(self.transitions)/self.fs
  
		self.scroll_widget = QtWidgets.QScrollBar(Qt.Horizontal)
		self.scroll_widget.setMinimum(self.plot_window_width)
		self.scroll_widget.setMaximum(len(self.time_ax)//self.fs)
		self.scroll_widget.setValue(self.plot_window_width)
		self.scroll_widget.valueChanged.connect(self.update_scroll)

		# find nearest transition
		candidates = (self.transitions <= self.scroll_widget.value()) & (self.transitions >= (self.scroll_widget.value()-self.plot_window_width))
		candidate = candidates.nonzero()[0]
		if len(candidate) > 0:
			candidate = candidate[0]
			txt = self.label_map[self.label_stream[int(candidate+1)*self.fs]]
			self.text = pg.TextItem(txt,color='black',anchor=(0,0))
			self.text.setPos(candidate, self.label_stream[int(candidate+1)*self.fs]+4)
			self.text.setParentItem(self.label_stream_widget_curve)
			self.text.setFont(self.my_font)

		# right side of GUI
		self.label_scroll_pane.addWidget(self.label_stream_widget)
		self.label_scroll_pane.addWidget(self.scroll_widget)

		self.right_vertical_pane.addLayout(self.plot_pane)
		self.right_vertical_pane.addLayout(self.label_scroll_pane)
		self.right_vertical_pane.setStretch(0,3)
		self.right_vertical_pane.setStretch(1,1)

		self.main_horizontal_pane.addLayout(self.right_vertical_pane)
		self.main_horizontal_pane.setStretch(0,1)
		self.main_horizontal_pane.setStretch(1,3)
		self.plot_pane.setSpacing(0)
		self.plot_pane.setContentsMargins(0,0,0,0)
		self.label_scroll_pane.setContentsMargins(0,0,56,0)

		widget = QWidget()
		widget.setLayout(self.main_horizontal_pane)
		self.setCentralWidget(widget)

		self.first_time = 0
		self.elapsed = 0
		self.pause_time = 0
		self.pause_elapsed = 0
		self.seen_candidates = [candidate]
		self.seen_packet_candidates = []

		self.packet_regions = {bp: {} for bp_i, bp in enumerate(self.body_parts)}

		self.last_was_time = False
		self.last_was_scroll = False

		self.clicked_start = 0
		self.clicked_stop = 0

		self.global_xmax = self.plot_window_width
		self.last_val = self.plot_window_width


	def update_plot_layout(self):
		val = self.scroll_widget.value()
		xmax = val
		xmin = xmax-self.plot_window_width
		clicked_bp = None
		for bp_i, bp in enumerate(self.body_parts):
			# add to checked if not already checked
			if self.checkboxes[bp].isChecked() and bp not in self.checked:
				clicked_bp = bp
				self.checked.append(bp)
				pw = pg.PlotWidget()
				pw.setLabel('left', bp)
				my_font = QFont("Times", 8, QFont.Bold)
				pw.getAxis("left").label.setFont(my_font)

				pw.showGrid(x = True, y = True)
				time_data = self.time_ax[int(xmin*self.fs):int(xmax*self.fs)]
				c1 = pw.plot(time_data,self.data_stream[bp_i*3,int(xmin*self.fs):int(xmax*self.fs)], pen={"color": "blue", "width": 1})
				c2 = pw.plot(time_data,self.data_stream[bp_i*3+1,int(xmin*self.fs):int(xmax*self.fs)], pen={"color": "orange", "width": 1})
				c3 = pw.plot(time_data,self.data_stream[bp_i*3+2,int(xmin*self.fs):int(xmax*self.fs)], pen={"color": "green", "width": 1})

				pw.getAxis('bottom').setStyle(tickLength=0, showValues=False)

				pw.setYRange(-20, 20)
				range_ = pw.getViewBox().viewRange() 
				pw.getViewBox().setLimits(yMin=range_[1][0], yMax=range_[1][1], minYRange = range_[1][1]-range_[1][0])

				# energy axis
				pw2 = pg.ViewBox()
				pw2.setYRange(min(self.e_plots[bp])-1e-5, max(self.e_plots[bp])+1e-5)
				pw.plotItem.showAxis('right')
				pw.plotItem.scene().addItem(pw2)
				pw.plotItem.getAxis('right').linkToView(pw2)
				pw2.setXLink(pw.plotItem)
				pw.plotItem.getAxis('right').setLabel('Energy', color='gray')

				curve = pg.PlotCurveItem(time_data,self.e_plots[bp][int(xmin*self.fs):int(xmax*self.fs)], pen='black')
				pw2.addItem(curve)
				
				self.plot_widgets[bp] = (pw,[c1,c2,c3],pw2,curve)
				self.plot_pane.addWidget(pw)

				self.update_views()
				pw.plotItem.vb.sigResized.connect(self.update_views)

				print("clicked", bp)


			# remove if not checked
			if not self.checkboxes[bp].isChecked() and bp in self.checked:
				self.checked.remove(bp)
				self.plot_pane.removeWidget(self.plot_widgets[bp][0])
				self.plot_widgets[bp][0].deleteLater()
				print("unclicked", bp)


		# find nearest packet
		print("updating lr",clicked_bp)
		all_ats = self.data_packets[clicked_bp][0]
		packet_candidates = (all_ats <= xmax) & (all_ats >= xmin)
		packet_candidates = packet_candidates.nonzero()[0]
		if len(packet_candidates) > 0:
			print(packet_candidates)
			for c in packet_candidates:
				packet_candidate = all_ats[c]
				if (packet_candidate,clicked_bp) in self.seen_packet_candidates:
					print("seen")
					if packet_candidate-16/self.fs < xmin:
						self.packet_regions[clicked_bp][packet_candidate].setRegion([xmin,packet_candidate])
					elif packet_candidate > xmax and packet_candidate-self.fs < xmax:
						self.packet_regions[clicked_bp][packet_candidate].setRegion([packet_candidate,xmax])
				else:
					# print("new packet:", packet_candidate, "bp: ", bp)
					
					self.seen_packet_candidates.append((packet_candidate,clicked_bp))
					pack = pg.LinearRegionItem([packet_candidate-16/self.fs,packet_candidate],movable=False,brush=(0, 0, 0, 50))
					if self.checkboxes[clicked_bp].isChecked():
						self.plot_widgets[clicked_bp][0].addItem(pack)
					self.packet_regions[clicked_bp][packet_candidate] = pack

		to_remove = []
		for pc in self.seen_packet_candidates:
			if self.checkboxes[pc[1]].isChecked():
				if pc[0] < xmin or pc[0] > xmax:
					# print("delete", pc,self.packet_regions[pc[1]][pc[0]])
					try:
						self.plot_widgets[pc[1]][0].removeItem(self.packet_regions[pc[1]][pc[0]])
					except:
						print("===========",pc)
						print(self.plot_widgets[pc[1]])
						exit()
					to_remove.append(pc)
		for pc in to_remove:
			self.seen_packet_candidates.remove(pc)

	def update_views(self):
		for bp_i, bp in enumerate(self.body_parts):
			# add to checked if not already checked
			if self.checkboxes[bp].isChecked():
				pw = self.plot_widgets[bp][0]
				pw2 = self.plot_widgets[bp][2]
				pw2.setGeometry(pw.plotItem.vb.sceneBoundingRect())
				pw2.linkedViewChanged(pw.plotItem.vb, pw2.XAxis)


	def update_scroll(self):

		# get scroll end point
		val = self.scroll_widget.value()

		xmax = val

		# offset right side by amount we let the timer go
		if self.last_was_time:
			# print("updating scroll",self.global_xmax)
			self.last_was_time = False
			if val > self.last_val:
				self.scroll_widget.setValue(int(self.global_xmax+1))
				xmax = int(self.global_xmax+1)
			else:
				self.scroll_widget.setValue(int(self.global_xmax+1))
				xmax = int(self.global_xmax)
		self.last_was_scroll = True
		xmin = xmax - self.plot_window_width

		# print(xmin,xmax,val,self.elapsed)
		
		time_data = self.time_ax[int(xmin*self.fs):int(xmax*self.fs)]
		self.label_stream_widget_curve.setData(time_data,self.label_stream[int(xmin*self.fs):int(xmax*self.fs)])
		for bp_i, bp in enumerate(self.body_parts):
			# add to checked if not already checked
			if self.checkboxes[bp].isChecked():
				for i in range(3):
					self.plot_widgets[bp][1][i].setData(time_data,self.data_stream[bp_i*3+i,int(xmin*self.fs):int(xmax*self.fs)])
				self.plot_widgets[bp][3].setData(time_data,self.e_plots[bp][int(xmin*self.fs):int(xmax*self.fs)])
		
		# find nearest transition
		candidates = (self.transitions <= xmax) & (self.transitions >= xmin)
		candidates = candidates.nonzero()[0]
		if len(candidates) > 0:
			for c in candidates:
				candidate = self.transitions[c]
				candidate = self.transitions[c]
				if candidate in self.seen_candidates:
					continue
				else:
					# print("new:", candidate)
					self.seen_candidates.append(candidate)
				txt = self.label_map[self.label_stream[int(candidate+1)*self.fs]]
				text = pg.TextItem(txt,color='black',anchor=(0,0))
				text.setPos(candidate, self.label_stream[int(candidate+1)*self.fs]+4)
				text.setParentItem(self.label_stream_widget_curve)
				text.setFont(self.my_font)

		# find nearest packet
		for bp_i, bp in enumerate(self.body_parts):
			if self.checkboxes[bp].isChecked():
				all_ats = self.data_packets[bp][0]
				packet_candidates = (all_ats <= xmax) & (all_ats >= xmin)
				packet_candidates = packet_candidates.nonzero()[0]
				if len(packet_candidates) > 0:
					for c in packet_candidates:
						packet_candidate = all_ats[c]
						if (packet_candidate,bp) in self.seen_packet_candidates:
							if packet_candidate-16/self.fs < xmin:
								self.packet_regions[bp][packet_candidate].setRegion([xmin,packet_candidate])
							elif packet_candidate > xmax and packet_candidate-self.fs < xmax:
								self.packet_regions[bp][packet_candidate].setRegion([packet_candidate,xmax])
						else:
							# print("new packet:", packet_candidate, "bp: ", bp)
							
							self.seen_packet_candidates.append((packet_candidate,bp))
							pack = pg.LinearRegionItem([packet_candidate-16/self.fs,packet_candidate],movable=False,brush=(0, 0, 0, 50))
							if self.checkboxes[bp].isChecked():
								self.plot_widgets[bp][0].addItem(pack)
							self.packet_regions[bp][packet_candidate] = pack

		to_remove = []
		for pc in self.seen_packet_candidates:
			if self.checkboxes[pc[1]].isChecked():
				if pc[0] < xmin or pc[0] > xmax:
					# print("delete", pc,self.packet_regions[pc[1]][pc[0]])
					try:
						self.plot_widgets[pc[1]][0].removeItem(self.packet_regions[pc[1]][pc[0]])
					except:
						print("===========",pc)
						print(self.plot_widgets[pc[1]])
						exit()
					to_remove.append(pc)
		for pc in to_remove:
			self.seen_packet_candidates.remove(pc)

		self.last_val = val


	def time_update(self):
		# get the current time
		now = perf_counter()
  
		xmax = self.global_xmax + (now - self.clicked_start)
		self.last_was_time = True

		xmin = xmax - self.plot_window_width
		
		time_data = self.time_ax[int(xmin*self.fs):int(xmax*self.fs)]
		self.label_stream_widget_curve.setData(time_data,self.label_stream[int(xmin*self.fs):int(xmax*self.fs)])
		
		for bp_i, bp in enumerate(self.body_parts):
			# add to checked if not already checked
			if self.checkboxes[bp].isChecked():
				for i in range(3):
					self.plot_widgets[bp][1][i].setData(time_data,self.data_stream[bp_i*3+i,int(xmin*self.fs):int(xmax*self.fs)])
				self.plot_widgets[bp][3].setData(time_data,self.e_plots[bp][int(xmin*self.fs):int(xmax*self.fs)])
		# find nearest transition
		candidates = (self.transitions <= xmax) & (self.transitions >= xmin)
		candidates = candidates.nonzero()[0]
		if len(candidates) > 0:
			for c in candidates:
				candidate = self.transitions[c]
				candidate = self.transitions[c]
				if candidate in self.seen_candidates:
					continue
				else:
					# print("new:", candidate)
					self.seen_candidates.append(candidate)
				txt = self.label_map[self.label_stream[int(candidate+1)*self.fs]]
				text = pg.TextItem(txt,color='black',anchor=(0,0))
				text.setPos(candidate, self.label_stream[int(candidate+1)*self.fs]+4)
				text.setParentItem(self.label_stream_widget_curve)
				text.setFont(self.my_font)

		# find nearest packet
		for bp_i, bp in enumerate(self.body_parts):
			if self.checkboxes[bp].isChecked():
				all_ats = self.data_packets[bp][0]
				packet_candidates = (all_ats <= xmax) & (all_ats >= xmin)
				packet_candidates = packet_candidates.nonzero()[0]
				if len(packet_candidates) > 0:
					for c in packet_candidates:
						packet_candidate = all_ats[c]
						if (packet_candidate,bp) in self.seen_packet_candidates:
							if packet_candidate-16/self.fs < xmin:
								self.packet_regions[bp][packet_candidate].setRegion([xmin,packet_candidate])
							elif packet_candidate > xmax and packet_candidate-self.fs < xmax:
								self.packet_regions[bp][packet_candidate].setRegion([packet_candidate,xmax])
						else:
							# print("new packet:", packet_candidate, "bp: ", bp)
							
							self.seen_packet_candidates.append((packet_candidate,bp))
							pack = pg.LinearRegionItem([packet_candidate-16/self.fs,packet_candidate],movable=False,brush=(0, 0, 0, 50))
							if self.checkboxes[bp].isChecked():
								self.plot_widgets[bp][0].addItem(pack)
							self.packet_regions[bp][packet_candidate] = pack

		to_remove = []
		for pc in self.seen_packet_candidates:
			if self.checkboxes[pc[1]].isChecked():
				if pc[0] < xmin or pc[0] > xmax:
					# print("delete", pc,self.packet_regions[pc[1]][pc[0]])
					try:
						self.plot_widgets[pc[1]][0].removeItem(self.packet_regions[pc[1]][pc[0]])
					except:
						print("===========",pc)
						print(self.plot_widgets[pc[1]])
						exit()
					to_remove.append(pc)
		for pc in to_remove:
			self.seen_packet_candidates.remove(pc)


	def start(self):
		self.last_was_time = True
		if self.last_was_scroll:
			self.global_xmax = self.scroll_widget.value()
			self.last_was_scroll  = False
		if self.pause_time != 0:
			self.pause_elapsed += perf_counter() - self.pause_time
		self.sbutton.setEnabled(False)
		self.ebutton.setEnabled(True)
		self.scroll_widget.setEnabled(False)
		self.timer.start(30)
		if self.first_time == 0:
			self.first_time = perf_counter()

		self.clicked_start = perf_counter()

	def stop(self):
		self.sbutton.setEnabled(True)
		self.ebutton.setEnabled(False)
		self.scroll_widget.setEnabled(True)
		self.timer.stop()
		self.pause_time = perf_counter()
		# print(self.scroll_widget.value(),self.elapsed/self.fs)
		self.clicked_stop = perf_counter()
		self.global_xmax += (self.clicked_stop - self.clicked_start)

	def closeEvent(self, event):
		self.timer.stop()
		event.accept()

	def prepare_data(self):
		pass



if __name__ == '__main__':
	app = QApplication(sys.argv)

	title = "IoTDI 2024"

	body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
	
	label_map = {
			 0:'sitting',
			 1:'standing',
			 2:'lying, back',
			 3:'lying, right side',
			 4:'ascending stairs',
			 5:'descending stairs',
			 6:'standing, elevator',
			 7:'moving, elevator',
			 8:'walking, parking lot',
			 9:'walking, flat treadmill',
			 10:'walking, inclined treadmill',
			 11:'running, treadmill',
			 12:'exercising, stepper',
			 13:'exercising, cross trainer',
			 14:'exercise bike horizontal',
			 15:'exercise bike vertical',
			 16:'rowing',
			 17:'jumping',
			 18:'playing basketball'
			 }
	
	label_stream = np.load('data_streams/val_labels.npy')
	data_stream = np.load('data_streams/val_data.npy')
	# time_ax = np.linspace(0,int(len(label_stream)/self.fs),len(label_stream))
	time_ax = np.arange(len(label_stream))/25
	t_axis = np.expand_dims(time_ax,axis=0)

	# add the time axis to the data
	full_data_window = np.concatenate([t_axis,data_stream],axis=0).T

	# energy harvesting parameters
	eh_params = {
		'proof_mass': 1*(10**-3),
		'spring_const': 0.17,
		'spring_damp': 0.0055,
		'disp_max': 0.01,
		'efficiency':0.3
	}
	eh = EnergyHarvester(**eh_params)

	# data_packets, e_plots, thresh = sparsify_data(full_data_window,body_parts,16,6e-6,eh,'opportunistic',visualize=True)
	# with open('packets.pickle', 'wb') as handle:
	# 	pickle.dump(data_packets, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# with open('e_plots.pickle', 'wb') as handle:
	# 	pickle.dump(e_plots, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# with open('thresh.pickle', 'wb') as handle:
	# 	pickle.dump(thresh, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('packets.pickle', 'rb') as handle:
		data_packets = pickle.load(handle)

	with open('e_plots.pickle', 'rb') as handle:
		e_plots = pickle.load(handle)

	with open('thresh.pickle', 'rb') as handle:
		thresh = pickle.load(handle)

	win = IoTDIDemo(body_parts, title, label_map, label_stream, data_stream, time_ax, data_packets, e_plots, thresh)

	win.show()
	sys.exit(app.exec_())