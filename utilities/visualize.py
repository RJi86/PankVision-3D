import os
import pydicom
import numpy as np
from mayavi import mlab
from tqdm import tqdm
import time

def read_dicom_images(patient_folder):
    dicom_slices = []
    total_slices = 0

    # Count the total number of DICOM files in the patient_folder
    for root, _, files in os.walk(patient_folder):
        for file in files:
            if file.endswith('.dcm'):
                total_slices += 1

    # Initialize the progress bar using tqdm
    with tqdm(total=total_slices, desc="Reading DICOM", unit="slice", leave=False) as pbar:
        for idx, (root, _, files) in enumerate(os.walk(patient_folder)):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_file_path = os.path.join(root, file)
                    start_time = time.time()  # Start time for reading each slice
                    dicom_slices.append(pydicom.dcmread(dicom_file_path))
                    end_time = time.time()  # End time after reading each slice
                    elapsed_time = end_time - start_time

                    print(f'Reading slice {idx + 1}/{total_slices}: {file}')
                    print('Time taken to read slice:', elapsed_time, 'seconds')

                    pbar.update(1)  # Update the progress bar

    dicom_slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))  # Sort slices by z-coordinate
    dicom_volume = np.stack([ds.pixel_array for ds in dicom_slices])
    spacings = [ds.PixelSpacing[0] for ds in dicom_slices]

    print("DICOM images read successfully for patient folder:", patient_folder)
    return dicom_volume, spacings

def visualize_3d_volume(volume):
    mlab.figure(bgcolor=(0, 0, 0), size=(800, 600))
    src = mlab.pipeline.scalar_field(volume)
    mlab.pipeline.volume(src)
    mlab.show()

# Specify the path to the patient's folder containing the DICOM images
patient_folder = '/Users/richardji/Library/CloudStorage/GoogleDrive-jirichard2007@gmail.com/My Drive/Machine-Learning-Biomedicine/Pancreatic-Cancer/data/Pancreas-CT/PANCREAS_0001'

# Read DICOM images and get the stacked CT volume
ct_volume, _ = read_dicom_images(patient_folder)

# Visualize the 3D CT volume using Mayavi
visualize_3d_volume(ct_volume)