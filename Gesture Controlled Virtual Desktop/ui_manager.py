from PyQt5 import QtWidgets, QtGui, QtCore
import sys

class GestureDesktopWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture-Controlled Virtual Desktop")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #222;")
        self.pen_color = QtCore.Qt.green
        self.active_tool = None
        self.drawing = False
        self.last_point = None

        self.canvas = QtGui.QPixmap(self.width(), self.height())
        self.canvas.fill(QtCore.Qt.transparent)

        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(self.canvas)
        self.label.setGeometry(0, 0, self.width(), self.height())

        self.gesture_label = QtWidgets.QLabel(self)
        self.gesture_label.setGeometry(10, 10, 300, 30)
        self.gesture_label.setStyleSheet("color: white; font-size: 18px;")

        self.show()

    def trigger_action(self, gesture):
        self.gesture_label.setText(f"Gesture: {gesture}")
        if gesture == "draw":
            self.active_tool = "draw"
        elif gesture == "tap":
            self.active_tool = None
        elif gesture == "swipe":
            self.clear_canvas()
        elif gesture == "pinch":
            self.pen_color = QtCore.Qt.red
        elif gesture == "drag":
            self.pen_color = QtCore.Qt.yellow

    def draw_point(self, x, y):
        if self.active_tool == "draw":
            painter = QtGui.QPainter(self.label.pixmap())
            pen = QtGui.QPen(self.pen_color, 5)
            painter.setPen(pen)

            if self.last_point:
                painter.drawLine(self.last_point, QtCore.QPoint(x, y))
            self.last_point = QtCore.QPoint(x, y)
            self.update()
        else:
            self.last_point = None


    def clear_canvas(self):
          self.canvas = QtGui.QPixmap(self.width(), self.height())
          self.canvas.fill(QtCore.Qt.transparent)
          self.label.setPixmap(self.canvas)
          self.last_point = None
