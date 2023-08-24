from controller_main import Controller
from model_main import Model
from view_main import View
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    model = Model()
    view = View()
    controller = Controller(model, view)
    controller.connect_signals()  # Connect the signals from the View to the slots in the Controller
    view.show()  # Display the main window
    sys.exit(app.exec_())

# from controller_main import Controller
# from model_main import Model
# from view_main import View
# from PyQt5.QtWidgets import QApplication
# import sys

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     model = Model()
#     view = View()
#     # view = View(controller)
#     controller = Controller(model, view)
#     sys.exit(app.exec_())