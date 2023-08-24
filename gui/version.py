# check if PyQt5 is install correctly
from PyQt5 import QtCore
print(QtCore.QT_VERSION_STR)

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
print(sys.version)
