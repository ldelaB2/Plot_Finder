from .gui import gui
from qgis.PyQt.QtWidgets import  QTabWidget

class optimization_gui(gui):
    def __init__(self, dlg, iface):
        super().__init__(dlg, iface)

    def init_gui(self):
        self.create_optimization_params(self.dlg.widget_optimization_params)
        self.connect_buttons()

    def load_gui(self):
        self.clear_params()
        self.load_input_layers(self.dlg.comboBox_img_input)
        self.load_input_layers(self.dlg.comboBox_plots_input)
        self.reroute_stdout()
        self.dlg.show()

    def connect_buttons(self):
        # Setting up the buttons
        self.dlg.pushButton_close.clicked.connect(self.close_windows)
        self.dlg.pushButton_viewlog.clicked.connect(self.disp_log)
        self.dlg.pushButton_run.clicked.connect(self.run_optimization)
        self.dlg.pushButton_output_path.clicked.connect(lambda: self.select_output_dir(self.dlg.lineEdit_output_path))
        self.dlg.pushButton_model_path.clicked.connect(lambda: self.select_file('model', self.dlg.lineEdit_model_path))

        # Set up checkbox
        self.dlg.checkBox_import_model.stateChanged.connect(self.toggle_import_model)

    def clear_params(self):
        self.dlg.comboBox_img_input.clear()
        self.dlg.comboBox_plots_input.clear()
        self.dlg.lineEdit_output_path.clear()
        self.logDialog.textEdit.clear()
        self.dlg.lineEdit_model_path.clear()
        tab_widget = self.dlg.findChild(QTabWidget)
        tab_widget.setCurrentIndex(0)
        
        #Turn things off to start
        self.dlg.lineEdit_model_path.setEnabled(False)
        self.dlg.pushButton_model_path.setEnabled(False)
        self.dlg.checkBox_import_model.setChecked(False)
        self.reset_params()

        
    def toggle_import_model(self):
        if self.dlg.checkBox_import_model.isChecked():
            self.dlg.lineEdit_model_path.setEnabled(True)
            self.dlg.pushButton_model_path.setEnabled(True)
        else:
            self.dlg.lineEdit_model_path.setEnabled(False)
            self.dlg.pushButton_model_path.setEnabled(False)

    def run_optimization(self):
        print("Starting Optimization")




