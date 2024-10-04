from .gui import gui
from .grayscale import create_gray_image
from qgis.PyQt.QtWidgets import  QTabWidget
from qgis.core import QgsProject

class plotfinder_gui(gui):
    def __init__(self, dlg, iface):
        super().__init__(dlg, iface)

    def init_gui(self):
        self.create_optional_params(self.dlg.scrollArea)
        self.create_optimization_params(self.dlg.widget_optimization_params)
        self.connect_bottons()

    def load_gui(self):
        self.clear_params()
        self.load_input_layers(self.dlg.comboBox_inputlayers)
        self.reroute_stdout()
        self.dlg.show()
        
    def connect_bottons(self):
        # Set up buttons
        self.dlg.pushButton_output.clicked.connect(lambda: self.select_output_dir(self.dlg.lineEdit_output))
        self.dlg.pushButton_close.clicked.connect(self.close_windows)
        self.dlg.pushButton_CheckParams.clicked.connect(self.check_params)
        self.dlg.pushButton_FindPlots.clicked.connect(self.plot_finder)
        self.dlg.pushButton_ViewGrayImg.clicked.connect(self.view_grayscale)
        self.dlg.pushButton_ViewLog.clicked.connect(self.disp_log)
        self.dlg.pushButton_paramfile.clicked.connect(lambda: self.select_file('paramfile', self.dlg.lineEdit_paramfile))
        self.dlg.pushButton_runparamimport.clicked.connect(self.run_paramimport)
        self.dlg.checkBox_grayscale.stateChanged.connect(self.toggle_grayscale)
        self.dlg.checkBox_optimization_model_import.stateChanged.connect(self.toggle_optimization_model)

        # Set up checkboxes
        self.dlg.checkBox_optimization.stateChanged.connect(self.toggle_optimization)
        self.dlg.checkBox_optional.stateChanged.connect(self.toggle_optional_params)
        self.dlg.checkBox_paramfile_import.stateChanged.connect(self.toggle_paramfile)

        # Set up combobox's
        self.dlg.comboBox_grayscale.addItems(["LAB","HSV", "BI", "SCI", "GLI","HI","NGRDI","SI","VARI","BGI","GREY"])
        self.dlg.comboBox_label_start.addItems(["Top Left", "Top Right", "Bottom Left", "Bottom Right"])
        self.dlg.comboBox_label_flow.addItems(["Snake","Linear"])


    def clear_params(self):
        # Turning things off to start
        self.dlg.lineEdit_paramfile.setEnabled(False)
        self.dlg.pushButton_paramfile.setEnabled(False)
        self.dlg.scrollArea.setEnabled(False)
        self.dlg.lineEdit_grayscale.setEnabled(False)
        self.dlg.pushButton_runparamimport.setEnabled(False)
        self.dlg.widget_optimization_params.setEnabled(False)

        self.logDialog.textEdit.clear()
        self.dlg.comboBox_inputlayers.clear()
        self.dlg.lineEdit_grayscale.clear()
        self.dlg.lineEdit_output.clear()
        self.dlg.checkBox_optional.setChecked(False)
        self.dlg.checkBox_optimization.setChecked(False)
        self.dlg.checkBox_paramfile_import.setChecked(False)
        self.dlg.checkBox_grayscale_invert.setChecked(False)
        self.dlg.checkBox_grayscale.setChecked(False)
        self.dlg.comboBox_label_start.setCurrentIndex(0)
        self.dlg.comboBox_grayscale.setCurrentIndex(0)
        self.dlg.comboBox_label_flow.setCurrentIndex(0)
        tab_widget = self.dlg.findChild(QTabWidget)
        tab_widget.setCurrentIndex(0)


        # Optimization Params
        self.dlg.checkBox_optimization_model_import.setChecked(False)
        self.dlg.checkBox_optimization_model_export.setChecked(False)
        self.dlg.checkBox_optimization_model_import.setEnabled(False)
        self.dlg.checkBox_optimization_model_export.setEnabled(False)
        self.dlg.lineEdit_optimization_model_import.clear()
        self.dlg.lineEdit_optimization_model_import.setEnabled(False)
        self.dlg.pushButton_optimization_model_import.setEnabled(False)
        self.dlg.widget_optimization_params.setEnabled(False)
        self.reset_params()


    def toggle_optimization(self):
        if self.dlg.checkBox_optimization.isChecked():
            self.dlg.widget_optimization_params.setEnabled(True)
            self.dlg.checkBox_optimization_model_import.setEnabled(True)
            self.dlg.checkBox_optimization_model_export.setEnabled(True)
        else:
            self.dlg.widget_optimization_params.setEnabled(False)
            self.dlg.checkBox_optimization_model_import.setEnabled(False)
            self.dlg.checkBox_optimization_model_export.setEnabled(False)

    def toggle_optional_params(self):
        if self.dlg.checkBox_optional.isChecked():
            self.dlg.scrollArea.setEnabled(True)
        else:
            self.dlg.scrollArea.setEnabled(False)

    def toggle_grayscale(self):
        if self.dlg.checkBox_grayscale.isChecked():
            self.dlg.comboBox_grayscale.setEnabled(False)
            self.dlg.lineEdit_grayscale.setEnabled(True)
        else:
            self.dlg.comboBox_grayscale.setEnabled(True)
            self.dlg.lineEdit_grayscale.setEnabled(False)

    def toggle_optimization_model(self):
        if self.dlg.checkBox_optimization_model_import.isChecked():
            self.dlg.lineEdit_optimization_model_import.setEnabled(True)
            self.dlg.pushButton_optimization_model_import.setEnabled(True)
        else:
            self.dlg.lineEdit_optimization_model_import.setEnabled(False)
            self.dlg.pushButton_optimization_model_import.setEnabled(False)

    def toggle_paramfile(self):
        if self.dlg.checkBox_paramfile_import.isChecked():
            self.dlg.lineEdit_paramfile.setEnabled(True)
            self.dlg.pushButton_paramfile.setEnabled(True)
            self.dlg.pushButton_runparamimport.setEnabled(True)
        else:
            self.dlg.lineEdit_paramfile.setEnabled(False)
            self.dlg.pushButton_paramfile.setEnabled(False)
            self.dlg.pushButton_runparamimport.setEnabled(False)


    def view_grayscale(self):
        if self.dlg.checkBox_grayscale.isChecked():
            method = self.dlg.lineEdit_grayscale.text()
            custom_flag = True
        else:
            method = self.dlg.comboBox_grayscale.currentText()
            custom_flag = False

        if self.dlg.checkBox_grayscale_invert.isChecked():
            invert_falg = True
        else:
            invert_falg = False

        #check if input layer is selected
        if not self.dlg.comboBox_inputlayers.currentText():
            print('No input layer selected')
            return
        else:
            input_layer = self.dlg.comboBox_inputlayers.currentText()
            print(f"Creating Grayscale Image from layer {input_layer} using {method} method")

        # Get the source file
        layers = QgsProject.instance().mapLayersByName(input_layer)
        layer = layers[0]
        source_file = layer.source()
       
        # Call the function
        create_gray_image(source_file, method, custom_flag, invert_falg)
        """
        # Add the grayscale image to the map
        layer_name = method + "Gray Scale"
        layer = QgsRasterLayer(gray_img_path, layer_name)
        if not layer.isValid():
            self.show_message('error', 'Invalid Layer')
            return
        
        QgsProject.instance().addMapLayer(layer)
        """

    def check_params(self):
        print("Checking Params")

    def plot_finder(self):
        print("Finding Plots")

    def run_paramimport(self):
        print("Running Param Import")

    
    


