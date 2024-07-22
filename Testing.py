import torch
from torch.utils.data import DataLoader
#from PretrainModel import model_smp as model
from Model import Model
from ClasseDataset import NeuronSegmentation
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

data_dir = "C:/Users/angel/OneDrive/Desktop/AI_fileAggiuntivi"

transforms = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


model = Model()
# Carica i pesi salvati
model.load_state_dict(torch.load("C:/Users/angel/OneDrive/Desktop/ARuntime/weights/weights.pt"))

# Sposta il modello sul dispositivo appropriato (CPU o GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_dataset = NeuronSegmentation(root_dir=data_dir, split='test', transforms=transforms)

test_loader = DataLoader(test_dataset, batch_size=4, num_workers=0, shuffle=False, drop_last=True)

def test_model(model, test_loader, device):
    model.eval()
    num_images_to_show = 15
    images_shown = 0

    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(images)  # Applicare sigmoid per normalizzare tra 0 e 1
            prediction = torch.argmax(outputs,1)

            # Crea la figura e la griglia di subplot
            fig, axs = plt.subplots(3, 1, figsize=(6, 18))

            # Immagine originale
            im = axs[0].imshow(TF.to_pil_image(make_grid((batch['image'][:,:,:,:]*255).cpu(), normalize=True, scale_each=True,padding=10,pad_value=25)))
            axs[0].axis('off')
            axs[0].set_title('Original Image')
            

            # Maschera
            mask = axs[1].imshow(TF.to_pil_image(make_grid(batch['mask'][:,:,:,:].cpu().float(), normalize=True, scale_each=True,padding=10,pad_value=25)))
            axs[1].axis('off')
            axs[1].set_title('Mask')
            

            # Predizione
            pred = axs[2].imshow(TF.to_pil_image(make_grid(prediction.unsqueeze(1)[:,:,:,:].cpu().float(), normalize=True, scale_each=True,padding=10,pad_value=25)))
            axs[2].axis('off')
            axs[2].set_title('Prediction')
            

            # Mostra la figura combinata
            plt.tight_layout()
            plt.show()


            images_shown += 1
            if images_shown >= num_images_to_show:
                return  


test_model(model, test_loader, device)