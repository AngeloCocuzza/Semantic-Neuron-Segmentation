
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


class NeuronSegmentation(Dataset):
    def __init__(self, root_dir, split='train', transforms=None, seed=42, val_frac=0.1, test_frac=0.1):
        super().__init__()

        random.seed(seed)
        self.root_dir = root_dir #Memorizza la directory principale del dataset come attributo della classe.
        img_names = os.listdir(os.path.join(root_dir, 'images')) #Legge i nomi di tutti i file nella directory images e li memorizza nella lista img_names
        random.shuffle(img_names) #Mescola casualmente i nomi delle immagini per assicurarsi che i dati siano distribuiti in modo casuale.
        num_val = int(len(img_names) * val_frac)
        num_test = int(len(img_names) * test_frac)
        num_train = len(img_names) - num_val - num_test

        #Calcola il numero di immagini da utilizzare per la validazione
        #(num_val), per il test (num_test) e per l'addestramento (num_train) in base alle frazioni specificate.

        if split == 'train':
            self.data = img_names[:num_train]
        elif split == 'val':
            self.data = img_names[num_train:num_train + num_val]
        elif split == 'test':
            self.data = img_names[-num_test:]
        else:
            raise ValueError('Valore di split non valido.')

        self.transforms = transforms #Memorizza le trasformazioni come attributo della classe.

    def __len__(self):
        return len(self.data) #Definisce il metodo __len__, che restituisce il numero di campioni nel dataset.

    def __getitem__(self, idx):
        #Definisce il metodo __getitem__, che viene chiamato per ottenere un singolo campione dal dataset.
        # Costruisce i percorsi completi delle immagini e delle maschere utilizzando l'indice idx.
        img_path = os.path.join(self.root_dir, 'images', self.data[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.data[idx])

        #Legge l'immagine e la maschera dai rispettivi percorsi. read_image è una funzione che legge l'immagine dal file,
        # e ImageReadMode.GRAY indica che la maschera viene letta in modalità scala di grigi.
        img = read_image(img_path)
        mask = read_image(mask_path, ImageReadMode.GRAY)

        #Crea un dizionario sample che contiene l'immagine e la maschera.
        sample = {'image': img, 'mask': mask}

        #Se sono state specificate delle trasformazioni, le applica sia all'immagine che alla maschera.
        if self.transforms:
            sample['image'] = self.transforms(sample['image'])
            sample['mask'] = self.transforms(sample['mask'])

        return sample #Restituisce il campione (l'immagine e la maschera) come output del metodo __getitem__.
 