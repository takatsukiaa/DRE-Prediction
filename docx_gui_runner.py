import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import Qt, Slot
import ui_docx_gui
import docx_parser_static

from ui_data_model import DocxTableModel


# The DocxApp class handles the business logic and signal-slot connections.

class DocxApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize the generated UI
        self.ui = ui_docx_gui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.parsed_data = {}
        # Connect signals to slots (event handling)
        self.init_signals()

    def init_signals(self):
        """
        Connects UI elements to their respective logic functions.
        """
        self.ui.btn_source.clicked.connect(self.handle_select_source)
        self.ui.btn_output.clicked.connect(self.handle_export_csv)
        self.ui.data_select.currentIndexChanged.connect(self.handle_selection_change)

    @Slot()
    def handle_select_source(self):
        """
        Logic for selecting the .docx file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Docx File", "", "Word Documents (*.docx)"
        )
        if file_path:
            # 1. Fetch raw data from parser
            raw_data = docx_parser_static.initial_table_selection(file_path)
            self.parsed_data = raw_data

            # 2. Dynamic Header Extraction
            headers = []
            if isinstance(raw_data, dict):
                headers = list(raw_data.keys())
            elif isinstance(raw_data, list) and len(raw_data) > 0:
                if isinstance(raw_data[0], dict):
                    headers = list(raw_data[0].keys())
                else:
                    # Fallback for list of lists: create generic headers
                    headers = [f"Field {i}" for i in range(len(raw_data[0]))]

            # 3. Instantiate and bind Model
            self.model = DocxTableModel(raw_data, headers)
            
            self.ui.parsed_table.setModel(self.model)
            self.ui.parsed_table.resizeColumnsToContents()

    @Slot()
    def handle_export_csv(self):
        """
        Logic for exporting data to CSV.
        """
        print("Exporting to CSV...")
        # Add your CSV conversion logic here
        
        if self.parsed_data:
            docx_parser_static.save_to_csv("output.csv")
            print(f"Data exported: {self.parsed_data}")
        else:
            print("No data to export.")

    @Slot(int)
    def handle_selection_change(self, index):
        """
        Handles the change in ComboBox selection.
        """
        selected_text = self.ui.data_select.itemText(index)
        print(f"Category changed to: {selected_text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = DocxApp()
    window.show()
    
    sys.exit(app.exec())