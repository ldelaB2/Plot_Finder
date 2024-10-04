from .gui import gui

class extraction_gui(gui):
    def __init__(self, dlg, iface):
        super().__init__(dlg, iface)

    def init_gui(self):
        self.connect_buttons()

    def load_gui(self):
        self.clear_params()
        self.load_input_layers(self.dlg.comboBox_img_input)
        self.load_input_layers(self.dlg.comboBox_plots_input)
        self.reroute_stdout()
        self.dlg.show()

    def connect_buttons(self):
        # Connect the buttons
        self.dlg.pushButton_close.clicked.connect(self.close_windows)
        self.dlg.pushButton_run.clicked.connect(self.run_plot_extraction)
        self.dlg.pushButton_view_log.clicked.connect(self.disp_log)
        self.dlg.pushButton_output_path.clicked.connect(lambda: self.select_output_dir(self.dlg.lineEdit_output_path))

        # Load the comboboxes
        self.dlg.comboBox_metadata.addItems(["true","false"])
        self.dlg.comboBox_output_file_type.addItems(["JPEG","TIF"])

    def clear_params(self):
        self.dlg.lineEdit_output_path.clear()
        self.logDialog.textEdit.clear()
        self.dlg.comboBox_metadata.setCurrentIndex(1)
        self.dlg.comboBox_output_file_type.setCurrentIndex(0)

    def run_plot_extraction(self):
        print("Extracting plots")
    
