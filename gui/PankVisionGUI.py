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
    QLayout,)
from monai.transforms import Compose, LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
from monai.networks.nets import UNet, DynUNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.utils import first
from monai.networks.blocks import activation
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
from monai.data import CacheDataset, DataLoader, Dataset
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap, QFont
import numpy as np
from skimage import exposure
from glob import glob
import nibabel as nib
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
sys.path.append('/Users/richardji/Desktop/PankVision-3D/model')
from get_model import get_model

class PancreaticSegmentationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_file_path = None
        self.volume = None
        self.segmentation = None
        # self.setFixedSize(1024, 768)

    def initUI(self):
        main_layout = QVBoxLayout()

        # Title
        font = QFont()
        title_label = QLabel("PankVision 3D Beta-1.0.1")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font.setFamily("Marion")
        font.setPointSize(24)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setStyleSheet("background-color: #d4ebd2; padding: 10px; min-width: 100%; border: 2px solid #5F9EA0;")

        # Create a QHBoxLayout for the title and images
        title_layout = QHBoxLayout()
        title_layout.setSizeConstraint(QLayout.SetFixedSize)
        
        # Add the title label to the layout
        title_layout.addWidget(title_label)

        # Images
        image1 = QLabel()
        pixmap1 = QPixmap('img/640px-PyTorch_logo_icon.svg.png')
        image1.setPixmap(pixmap1.scaled(50, 50, Qt.KeepAspectRatio))

        image2 = QLabel()
        pixmap2 = QPixmap('img/monai.png')
        image2.setPixmap(pixmap2.scaled(50, 50, Qt.KeepAspectRatio))

        # Add the images to the layout
        title_layout.addWidget(image1)
        title_layout.addWidget(image2)

        # Add the title layout to the main layout
        main_layout.addLayout(title_layout)

        # Horizontal line after the title
        title_line = QFrame()
        title_line.setFrameShape(QFrame.HLine)
        title_line.setFrameShadow(QFrame.Sunken)

        main_layout.addWidget(title_line)

        self.setLayout(main_layout)

        navigation_layout = QHBoxLayout()
        navigation_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Input section
        input_layout = QVBoxLayout()
        input_layout.setSizeConstraint(QLayout.SetFixedSize)

        input_label = QLabel("Input Volume")
        input_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_label.setStyleSheet("color: black")
        input_layout.addWidget(input_label)

        self.select_file_button = QPushButton("Import file (NIFTI/DCM)")
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

        output_label = QLabel("Slice Changer")
        output_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_layout.addWidget(output_label)

        # Remove the fixed minimum and maximum values
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow it to expand

        # Change the slider handle shape to a rectangle and a square
        self.slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #5F9EA0;  /* Color of the handle */
                border: 1px solid #3C7F7E; /* Border color of the handle */
                width: 20px;  /* Width of the rectangle */
                height: 10px; /* Height of the square */
                border-radius: 2px; /* Rounded corners */
            }
        """)

        self.slider.valueChanged.connect(lambda value: self.update_slice_number(value))
        output_layout.addWidget(self.slider)

        # Initialize the label with the current slice number (0)
        self.slice_number_label = QLabel()
        self.slice_number_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow it to expand
        output_layout.addWidget(self.slice_number_label)
        self.update_slice_number()  # Set the initial label text to "Slice Number: 0"

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

        # Create a QHBoxLayout to display the original CT slice and segmentation image side by side
        image_and_segmentation_layout = QHBoxLayout()
        image_and_segmentation_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Create a black square placeholder for the original CT scan
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(512, 512)  # Set the size to 512x512 pixels
        self.original_label.setStyleSheet("background-color: black;")

        # Create a black square placeholder for the segmentation image
        self.segmented_label = QLabel()
        self.segmented_label.setAlignment(Qt.AlignCenter)
        self.segmented_label.setFixedSize(512, 512)  # Set the size to 512x512 pixels
        self.segmented_label.setStyleSheet("background-color: black;")

        # Add both labels to the horizontal layout
        image_and_segmentation_layout.addWidget(self.original_label)
        image_and_segmentation_layout.addWidget(self.segmented_label)

        # Add the horizontal layout to the image layout
        image_layout.addLayout(image_and_segmentation_layout)

        main_layout.addLayout(image_layout)

        # Initialize slice
        self.slice_label = QLabel()

        self.setLayout(main_layout)

        self.setWindowTitle('PankVision 3D 1.0.1 Demo')
        self.show()

    def update_slice_number(self, value=None):
        if hasattr(self, 'volume') and self.volume is not None:
            if value is None:
                # If value is None, set it to 0 (or any default value)
                value = 0
            slice_number = int(value)  # Ensure value is an integer
            if 0 <= slice_number < self.volume.shape[2]:
                volume_slice = self.volume[:, :, slice_number]
                # Normalize data to 0-255 range for QImage
                volume_slice = (volume_slice - np.min(volume_slice)) / (np.max(volume_slice) - np.min(volume_slice)) * 255
                volume_slice = volume_slice.astype(np.uint8)
                height, width = volume_slice.shape
                qimage = QImage(
                    volume_slice.data.tobytes(),  # Convert to bytes
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8
                )
                pixmap = QPixmap.fromImage(qimage)
                self.original_label.setPixmap(pixmap)
                self.slice_number_label.setText(f"Slice Number: {slice_number}")

    def preprocess_image(self, image, a_min=-200, a_max=200):
        adjusted_image = (image - 60 + 200) / 400
        normalized_image = (adjusted_image - np.min(adjusted_image)) / (np.max(adjusted_image) - np.min(adjusted_image))
        # convert the PyTorch tensor to a NumPy array
        normalized_image_np = normalized_image.numpy()
        enhanced_image = exposure.equalize_adapthist(normalized_image_np)
        return enhanced_image

    def prepare_test(self, in_file, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=False):
        def combined_transforms(data):
            enhanced_vol = self.preprocess_image(data["vol"], a_min=a_min, a_max=a_max)
            return {"vol": enhanced_vol}

        test_files = [{"vol": in_file}]

        combined_test_transforms = Compose(
            [
                LoadImaged(keys=["vol"]),
                AddChanneld(keys=["vol"]),
                Spacingd(keys=["vol"], pixdim=pixdim, mode=("bilinear")),
                Orientationd(keys=["vol"], axcodes="RAS"),
                ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=['vol'], source_key='vol'),
                Resized(keys=["vol"], spatial_size=spatial_size),
                # Applying the combined preprocessing function here
                combined_transforms,
                ToTensord(keys=["vol"]),
            ]
        )

        test_ds = Dataset(data=test_files, transform=combined_test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return test_loader
    
    def select_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select NIfTI File', filter="NIfTI files (*.nii *.nii.gz);;DICOM files (*.dcm)")
        if file_name[0]:
            self.selected_file_path = str(file_name[0])

            try:
                nifti_image = nib.load(self.selected_file_path)
                self.volume = nifti_image.get_fdata()
                num_slices = self.volume.shape[2]
                print("Loaded NIfTI data shape:", self.volume.shape)
                print("Min value:", np.min(self.volume))
                print("Max value:", np.max(self.volume))

                # Update the maximum value of the slider to match the number of slices
                self.slider.setMinimum(0)
                self.slider.setMaximum(num_slices - 2)
                self.slider.setSingleStep(1)  # Set the step size for the slider

                # Reset the slider to the first slice
                self.slider.setValue(0)

            except Exception as e:
                print(f"Error loading NIfTI file: {str(e)}")
                return

            # Load and display the segmentation results
            self.load_segmentation_results()

            # Display the first slice
            self.update_slice_number()

    def load_segmentation_results(self):
        # Load your segmentation model and perform inference here
        # Replace this part with your code
        device = torch.device("cpu")
        args = {
            'model_name': 'DynUNet',
            'pretrained': True,
            'dropout': 0.1
        }
        model = get_model(args)
        model = model.to(device)
        model.load_state_dict(torch.load("/Users/richardji/Library/CloudStorage/GoogleDrive-jirichard2007@gmail.com/My Drive/Machine-Learning-Biomedicine/PankVision-3D/results/dataset-007/v6dynunet/best_metric_model.pth", map_location=device))
        model.eval()
        
        sw_batch_size = 4
        roi_size = (128, 128, 64)

        with torch.no_grad():
            # Convert self.volume to a PyTorch tensor
            # volume_tensor = torch.tensor(self.volume, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
            test_loader = self.prepare_test(self.selected_file_path)
            test_patient = first(test_loader)
            t_volume = test_patient['vol']
            # test_outputs = sliding_window_inference(volume_tensor, roi_size, sw_batch_size, model)
            test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
            sigmoid_activation = torch.sigmoid(test_outputs)
            test_outputs = sigmoid_activation > 0.53
            
            # Ensure the test_outputs is of data type np.uint8
            test_outputs_np = test_outputs[0, 1, ...].detach().cpu().numpy().astype(np.uint8)

            # Resize the segmentation output to match the original CT scan size
            original_size = self.volume.shape
            segmentation_resized = cv2.resize(test_outputs_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

            # Store the segmentation results in self.segmentation
            self.segmentation = segmentation_resized
            print("Segmentation Output Size:", self.segmentation.shape)

            # Now, update both the original and segmented images for the currently selected slice
            self.update_slice_number(self.slider.value())  # Update the displayed slice

    def update_slice_number(self, value=None):
        if hasattr(self, 'volume') and self.volume is not None and hasattr(self, 'segmentation') and self.segmentation is not None:
            if value is None:
                value = 0
            slice_number = int(value)
            if (
                0 <= slice_number < self.volume.shape[2] and
                0 <= slice_number < self.segmentation.shape[0]  # Adjusted for the resized segmentation
            ):
                volume_slice = self.volume[:, :, slice_number]
                segmentation_slice = self.segmentation[:, :, slice_number]  # Extract the resized segmentation slice

                volume_slice = (volume_slice - np.min(volume_slice)) / (np.max(volume_slice) - np.min(volume_slice)) * 255
                volume_slice = volume_slice.astype(np.uint8)

                segmentation_slice = (segmentation_slice > 0.5).astype(np.uint8) * 255

                height, width = volume_slice.shape
                qimage_volume = QImage(
                    volume_slice.data.tobytes(),
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8
                )
                pixmap_volume = QPixmap.fromImage(qimage_volume)

                qimage_segmentation = QImage(
                    segmentation_slice.data.tobytes(),
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8
                )
                pixmap_segmentation = QPixmap.fromImage(qimage_segmentation)

                self.original_label.setPixmap(pixmap_volume)
                self.segmented_label.setPixmap(pixmap_segmentation)
                self.slice_number_label.setText(f"Slice Number: {slice_number}")

    def update_figure(self):
        slice_number = self.slider.value()
        slice_data = self.segmentation_results[:,:,slice_number]
        height, width = slice_data.shape
        bytes_per_line = width
        # Convert the MetaTensor to a numpy array before calling tobytes
        image = QImage(slice_data.detach().cpu().numpy().tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        self.slice_label.setPixmap(pixmap)

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = PancreaticSegmentationGUI()
        window.setStyleSheet("background-color: #eeeeee;")
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred: {str(e)}") 