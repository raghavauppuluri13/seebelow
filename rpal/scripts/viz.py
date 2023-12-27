import sys

from PyQt5 import Qt
from vedo import Cone, Plotter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.widget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.plt = Plotter(N=2, axes=1, qt_widget=self.widget)
        self.id1 = self.plt.add_callback("mouse click", self.onMouseClick)
        self.id2 = self.plt.add_callback("key press", self.onKeypress)

        self.dataset_path = dataset_path
        self.data_file_path = data_file_path
        self.data = np.loadtxt(data_file_path)
        self.plot_data_position = [[], [], []]
        self.plot_data_force = [[], [], []]
        self.current_index = 0

        # Set up the rest of the Qt window
        self.layout.addWidget(self.widget)
        self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()  # NB: qt, not a Plotter method

    def onMouseClick(self, evt):
        print("mouse clicked")

    def onKeypress(self, evt):
        print("key pressed:", evt.keypress)

    @Qt.pyqtSlot()
    def onClick(self):
        self.plt.actors[0].color("red5").rotate_z(40)
        self.plt.render()

    def onClose(self):
        self.widget.close()


if __name__ == "__main__":

    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()
