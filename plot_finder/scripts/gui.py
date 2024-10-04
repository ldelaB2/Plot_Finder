from qgis.core import Qgis, QgsProject
from qgis.PyQt.QtWidgets import QScrollArea, QTabWidget, QSpinBox, QComboBox, QLineEdit, QTextEdit, QDialog, QFileDialog, QGridLayout, QPushButton, QWidget, QGridLayout, QLabel, QVBoxLayout
from qgis.PyQt.QtCore import QObject, pyqtSignal
import sys, os


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class LogDialog(QDialog):
    def __init__(self, parent=None):
        """Constructor."""
        super(LogDialog, self).__init__(parent)
        self.setWindowTitle("Log")
        #Create the text box
        self.textEdit = QTextEdit(self)
        # Create the close button
        self.closeButton = QPushButton("Close", self)
        self.closeButton.clicked.connect(self.close)

        layout = QGridLayout(self)
        layout.addWidget(self.textEdit, 0, 0, 1, 3)
        layout.addWidget(self.closeButton, 1, 2)
        self.setLayout(layout)
        
        # Set the size of the log dialog
        self.resize(600, 300)

class gui:
    def __init__(self, dlg, iface):
        self.dlg = dlg
        self.iface = iface
        self.logDialog = LogDialog()
        self.load_params()
        self.param_dict = {}

    def load_params(self):
        self.optional_params = [
                            ["Number of Cores:", 'spin', [1, os.cpu_count()], os.cpu_count()],
                            ["QC Depth:", 'combo', ['min','max','none'], 'min'],
                            ["Box Radius Height:", 'spin', [0,5000], 800],
                            ["Box Radius Width:", 'spin', [0,5000], 500],
                            ["Sparse Skip Size:", 'spin', [0,1000], 100],
                            ["Frequency Filter Width:", 'spin', [0,100], 1],
                            ["Row Sig Remove:", 'spin', [0,20], 2],
                            ["Num Sig Returned:", 'spin', [0,20], 2],
                            ["Expand Radi:", 'spin', [0,20], 2],
                            ["Wave Pixel Expand:", 'spin', [0,20], 0],
                            ["Poly Deg Range:", 'spin', [0,10], 3],
                            ["Poly Deg Row:", 'spin', [0,10], 1],
                            ["Extract Plots:", 'combo', ['true','false'], 'false'],
                            ["Create Shapefile:", 'combo', ['true','false'], 'true']
        ]

        general_params = [
                            ["General"],
                            ["Optimization Meta Miter:", 'spin', [0,10], 3],
                            ["Optimization X Radi:", 'spin', [0,500], 20],
                            ["Optimization Y Radi:", 'spin', [0,500], 20],
                            ["Optimization T Radi:", 'spin', [0,180], 5],
                            ["Method", 'combo', ['PSO','Genetic','Anneal'], 'PSO']
        ]

        pso_params = [      
                            ["PSO"],
                            ["PSO Particles:", 'spin', [0,100], 10],
                            ["PSO C1:", 'spin', [0,5], 2],
                            ["PSO C2:", 'spin', [0,5], 2],
                            ["PSO W:", 'spin', [0,5], 1]
        ]

        genetic_params = [
                            ["Genetic"],
                            ["Genetic Population:", 'spin', [0,100], 10],
                            ["Genetic Crossover:", 'spin', [0,1], 1],
                            ["Genetic Mutation:", 'spin', [0,1],1]
        ]

        anneal_params = [
                            ["Anneal"],
                            ["Anneal T0:", 'spin', [0,100], 50],
                            ["Anneal T1:", 'spin', [0,100], 1]
        ]

        self.optomization_parms = [general_params, pso_params, genetic_params, anneal_params]

    def reroute_stdout(self):
        # Redirect stdout and stderr to the log dialog
        self.original_std_out = sys.stdout
        self.original_std_err = sys.stderr
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

    def show_message(self, type, message):
        if type == 'error':
            self.iface.messageBar().pushMessage("Error", message, level=Qgis.Critical, duration=3)
        elif type == 'warning':
            self.iface.messageBar().pushMessage("Warning", message, level=Qgis.Warning, duration=3)
        elif type == 'info':
            self.iface.messageBar().pushMessage("Info", message, level=Qgis.Info, duration=3)
        elif type == 'success':
            self.iface.messageBar().pushMessage("Success", message, level=Qgis.Success, duration=3)
        else:
            print("Invalid message type")

    def select_output_dir(self, lineEdit):
        output_folder = str(QFileDialog.getExistingDirectory(self.dlg, "Select Directory"))
        if not output_folder:
            if not lineEdit.text():
                print("No output directory selected")
        else:
            lineEdit.setText(output_folder)
            print(f"Saving output to {output_folder}")

    def close_windows(self):
        self.dlg.close()
        self.logDialog.close()
        sys.stdout = self.original_std_out
        sys.stderr = self.original_std_err

    def onUpdateText(self, text):
        cursor = self.logDialog.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.logDialog.textEdit.setTextCursor(cursor)
        self.logDialog.textEdit.ensureCursorVisible()

    def select_file(self, type, lineEdit):
        if type == 'model':
            file = QFileDialog.getOpenFileName(self.dlg, "Select JPG File", "", "JPG Files (*.jpg)")[0]
        elif type == 'shapefile':
            file = QFileDialog.getSaveFileName(self.dlg, "Select Shapefile", "", "Shapefile (*.shp)")[0]
        elif type == 'paramfile':
            file = QFileDialog.getOpenFileName(self.dlg, "Select Param File", "", "Param Files (*.json)")[0]
        else:
            print("Invalid File Type")
            return

        if not file:
            if not lineEdit.text():
                print("No file selected")
        else:
            lineEdit.setText(file)
            print(f"Selected file {file}")

    def disp_log(self):
        self.logDialog.show()

    def load_input_layers(self, comboBox):
        # Featch currently loaded layers
        layers = QgsProject.instance().layerTreeRoot().children()
        # Add the layers to the img combbox 
        comboBox.addItems([layer.name() for layer in layers])

    def create_optional_params(self, scroll_area):
        # Create the widget
        scrollwidget = QWidget()
        scrolllayout = QVBoxLayout(scrollwidget)
        
        # Fill in the widget
        for param in self.optional_params:
            rowWidget = self.create_grid_layout(param)
            self.param_dict[param[0]] = param[3]
            scrolllayout.addWidget(rowWidget)

        # Set the scroll area
        scroll_area.setWidget(scrollwidget)    

    def create_optimization_params(self, frame):
        tab_widget = QTabWidget()
        frame_layout = QVBoxLayout(frame)

        for params in self.optomization_parms:
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)

            for param in params[1:]:
                rowWidget = self.create_grid_layout(param)
                self.param_dict[param[0]] = param[3]
                param_layout.addWidget(rowWidget)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(param_widget)

            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(scroll_area)
            tab.setLayout(layout)

            tab_widget.addTab(tab, params[0][0])

        frame_layout.addWidget(tab_widget)
        frame.setLayout(frame_layout)

    def reset_params(self):
        for param_name, defaut_vale in self.param_dict.items():
            widget = self.dlg.findChild(QWidget, param_name)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(defaut_vale)
            elif isinstance(widget, QSpinBox):
                widget.setValue(defaut_vale)
            else:
                print("Invalid Param Type")

    def create_grid_layout(self, param_list):
            param = param_list[0]
            param_type = param_list[1]
            param_values = param_list[2]
            param_default = param_list[3]
            
            rowWidget = QWidget()
            gridlayout = QGridLayout(rowWidget)

            gridlayout.addWidget(QLabel(param), 0, 0)
            # Check the param type
            if param_type == 'combo':
                combo = QComboBox()
                combo.setObjectName(param)
                combo.addItems(param_values)
                combo.setCurrentText(param_default)
                gridlayout.addWidget(combo, 0, 1)
            elif param_type == 'spin':
                spin = QSpinBox()
                spin.setObjectName(param)
                spin.setMinimum(param_values[0])
                spin.setMaximum(param_values[1])
                spin.setValue(param_default)
                gridlayout.addWidget(spin, 0, 1)
            else:
                print("Invalid Param Type")

            rowWidget.setFixedHeight(40)

            return rowWidget

    

