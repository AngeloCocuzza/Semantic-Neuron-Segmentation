import numpy as np
import os
import glob
from PIL import Image

#Caricamento da file .npy ,estrazione di immagini e array di maschere 
def load_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True).item()
        max_projection = data.get('max_projection', None)
        roi_mask_array = data.get('roi_mask_array', None)

        if max_projection is None or roi_mask_array is None:
            print(f"Warning: Missing data in file {file_path}")
            return None, None

        # Combina tutte le ROI masks in una singola mask
        combined_mask = np.bitwise_or.reduce(roi_mask_array, axis=0)

        return max_projection, combined_mask

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None
    
#Suddivisioni dell'immagine in blocchi piu piccoli con dimensione patch size
def split_image(image, mask, patch_size):
    patches_img = []
    patches_mask = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patch_img = image[i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]
            if patch_img.shape == (patch_size, patch_size) and patch_mask.shape == (patch_size, patch_size):
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
    return patches_img, patches_mask


def prepare_dataset(data_dir, patch_size):
    images = []
    masks = []

    for file_path in glob.glob(os.path.join(data_dir, '*.npy')):
        image, mask = load_data(file_path)
        if image is not None and mask is not None:
            if image.shape != (512, 512):
                continue  

            
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)  

        
            patches_img, patches_mask = split_image(image, mask, patch_size)
            images.extend(patches_img)
            masks.extend(patches_mask)

    return np.array(images), np.array(masks)


data_dir = "C:/Users/angel/OneDrive/Desktop/AI_fileAggiuntivi"
patch_size = 256

# carica il dataset
images, masks = prepare_dataset(data_dir, patch_size)

# Controlla se le imaggini e le maschere sono caricate correttamente
print(f"Number of images loaded: {len(images)}")
print(f"Number of masks loaded: {len(masks)}")


if len(images) == 0 or len(masks) == 0:
    raise ValueError("No images or masks were loaded. Check the data directory and file contents.")

# Crea delle cartelle per salvare immagini e maschere in formato JPG
output_image_dir = os.path.join(data_dir, 'images')
output_mask_dir = os.path.join(data_dir, 'masks')

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Salva imaggini e maschere
for i, (image, mask) in enumerate(zip(images, masks)):
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)

    image_pil.save(os.path.join(output_image_dir, f'image_{i}.jpg'))
    mask_pil.save(os.path.join(output_mask_dir, f'image_{i}.jpg'), mode='L')

print("Images and masks have been saved as JPG files.")

