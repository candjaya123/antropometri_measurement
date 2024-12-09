from PyQt5.QtCore import QFile

class AppStyle:
    def __init__(self, filename:str):
        self.filename = filename
        self.__load_style()

    def __load_style(self):
        css_file = QFile(self.filename)
        css_file.open(QFile.ReadOnly | QFile.Text)
        self.stylesheet = css_file.readAll().data().decode("utf-8")