from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase

# Create a QApplication instance
app = QApplication([])

# Create a QFontDatabase instance
font_database = QFontDatabase()

# Get the list of all available font families
font_families = font_database.families()

# Print each font family
for font in font_families:
    print(font)
