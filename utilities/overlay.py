import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
import os

path_img = '/Users/richardji/Library/CloudStorage/GoogleDrive-jirichard2007@gmail.com/My Drive/Machine-Learning-Biomedicine/Pancreatic-Cancer/pancreas/nifti_files/images'
path_lab = '/Users/richardji/Library/CloudStorage/GoogleDrive-jirichard2007@gmail.com/My Drive/Machine-Learning-Biomedicine/Pancreatic-Cancer/pancreas/nifti_files/labels'

# Load the first nifti image
img = 'pancreas_001_0.nii.gz'
path_i = os.path.join(path_img, img)
final_img = nib.load(path_i)
data1 = final_img.get_fdata()

# Load the second nifti image
path_l = os.path.join(path_lab, img)
final_lab = nib.load(path_l)
data2 = final_lab.get_fdata()

num_slices = data1.shape[2]

# Create a figure and axes
fig, ax = plt.subplots()

# Set the initial slice to display
current_slice = 0

# Function to update the displayed slice
def update_slice(val):
    # Get the current slice from the slider value
    current_slice = int(val)
    
    # Clear the current plot
    ax.clear()
    
    # Plot the first image
    ax.imshow(data1[:,:,current_slice], cmap='gray')
    
    # Overlay the second image on top of the first image
    ax.imshow(data2[:,:,current_slice], cmap='jet', alpha=0.5)

    # Update the figure title
    ax.set_title(f'Slice {current_slice+1} of {num_slices}')
    

    fig.canvas.draw()

# Create a slider to control the current slice
ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
slider = Slider(ax_slider, 'Slice', 0, num_slices-1, valinit=0, valstep=1)
slider.on_changed(update_slice)

update_slice(0)

plt.show()

