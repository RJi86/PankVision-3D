import sys
from PyQt5.QtWidgets import(
    QApplication, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QPushButton, 
    QFileDialog, 
    QSlider, 
    QFrame,
    QSizePolicy,
    QLayout)
from monai.transforms import Compose, LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.utils import first
from monai.networks.blocks import activation
import torch
import matplotlib.pyplot as plt
import os
from monai.data import CacheDataset, DataLoader, Dataset
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
import numpy as np

class PancreaticSegmentationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # self.setFixedSize(1024, 768)

    def initUI(self):
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Pancreatic Segmentation Tool Demo")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        main_layout.addWidget(title_label)

        # Horizontal line after the title
        title_line = QFrame()
        title_line.setFrameShape(QFrame.HLine)
        title_line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(title_line)

        navigation_layout = QHBoxLayout()
        navigation_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Input section
        input_layout = QVBoxLayout()
        input_layout.setSizeConstraint(QLayout.SetFixedSize)

        input_label = QLabel("Input")
        input_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_label.styleSheet("color: white")
        input_layout.addWidget(input_label)

        self.select_file_button = QPushButton("Import file (NIFTI)")
        self.select_file_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.select_file_button.clicked.connect(self.select_file)
        input_layout.addWidget(self.select_file_button)

        navigation_layout.addLayout(input_layout)

        # Add a vertical line as a barrier
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        navigation_layout.addWidget(line)

        # Output section
        output_layout = QVBoxLayout()
        output_layout.setSizeConstraint(QLayout.SetFixedSize)

        output_label = QLabel("Output")
        output_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_layout.addWidget(output_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.slider.setMinimum(0)
        self.slider.setMaximum(64)
        self.slider.valueChanged.connect(self.update_slice_number)
        output_layout.addWidget(self.slider)

        self.slice_number_label = QLabel("Slice Number: 0")
        self.slice_number_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_layout.addWidget(self.slice_number_label)

        navigation_layout.addLayout(output_layout)

        main_layout.addLayout(navigation_layout)

        # Divider after the output section
        divider_line = QFrame()
        divider_line.setFrameShape(QFrame.HLine)
        divider_line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(divider_line)

        # Image and Segmentation section
        image_layout = QVBoxLayout()
        image_layout.setSizeConstraint(QLayout.SetFixedSize)
        image_label = QLabel("Image and Segmentation")
        image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        image_layout.addWidget(image_label)

        # Create a QLabel to display the image and masks
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.original_label)

        self.segmented_label = QLabel()
        self.segmented_label.setAlignment(Qt.AlignCenter)
        self.segmented_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.segmented_label)

        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

    def update_slice_number(self):
        slice_number = self.slider.value()
        self.slice_number_label.setText(f"Slice Number: {slice_number}")

    def select_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select NIfTI File', filter="NIfTI files (*.nii *.nii.gz)")
        if file_name[0]:
            self.selected_file_path = file_name[0]  # Store the selected file path for later use

            # Preprocess the selected NIfTI file
            test_files = [{"vol": file_name[0], "seg": ""}]
            test_transforms = Compose(
                [
                    LoadImaged(keys=["vol", "seg"]),
                    AddChanneld(keys=["vol", "seg"]),
                    Spacingd(keys=["vol", "seg"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear", "nearest")),
                    Orientationd(keys=["vol", "seg"], axcodes="RAS"),
                    ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
                    Resized(keys=["vol", "seg"], spatial_size=[128, 128, 64]),
                    ToTensord(keys=["vol", "seg"]),
                ]
            )
            test_ds = Dataset(data=test_files, transform=test_transforms)
            test_loader = DataLoader(test_ds)

            # Perform segmentation on the preprocessed data
            device = torch.device("cuda:0")
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=0.2
            ).to(device)

            model.load_state_dict(torch.load("best_metric_model.pth"))
            model.eval()

            # Display the segmentation results
            sw_batch_size = 4
            roi_size = (128, 128, 64)
            with torch.no_grad():
                test_patient = first(test_loader)
                t_volume = test_patient['vol']

                test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
                sigmoid_activation = activation(sigmoid=True)
                test_outputs = sigmoid_activation(test_outputs)
                test_outputs = test_outputs > 0.53

                # Store the segmentation results in a variable
                self.segmentation_results = test_outputs.detach().cpu()[0][1]

                # Update the displayed figure
                self.update_figure()

    def update_figure(self):
        slice_number = self.slider.value()
        slice_data = self.segmentation_results[:,:,slice_number]
        height, width = slice_data.shape
        bytes_per_line = width
        image = QImage(slice_data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        self.slice_label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PancreaticSegmentationGUI()
    window.setStyleSheet("background-color: #23408e;")
    window.show()
    sys.exit(app.exec_())