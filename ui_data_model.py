from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

class DocxTableModel(QAbstractTableModel):
    def __init__(self, data=None, headers=None):
        super().__init__()
        # Data Normalization: Ensure self._data is always a list
        if isinstance(data, dict):
            # If a single dictionary is passed, wrap it in a list to represent one row
            self._data = [data]
        elif isinstance(data, list):
            self._data = data
        else:
            self._data = []
        
        self._headers = headers or []

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._data)):
            return None

        if role == Qt.DisplayRole:
            row_data = self._data[index.row()]
            col_key = self._headers[index.column()]

            # Duck Typing: Handle both dictionary and list-based rows
            if isinstance(row_data, dict):
                return str(row_data.get(col_key, ""))
            elif isinstance(row_data, (list, tuple)):
                return str(row_data[index.column()]) if index.column() < len(row_data) else ""
        
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self._headers[section]) if section < len(self._headers) else f"Column {section}"
        return None