import torch
import multiprocessing
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms as T
from ClasseDataset import NeuronSegmentation
import torch.nn as nn
import torch.nn.functional as F
#from PretrainModel import model_smp as model
from Training import train
from Model import Model


data_dir = "C:/Users/angel/OneDrive/Desktop/AI_fileAggiuntivi"

torch.cuda.empty_cache()

# Seleziona il dispositivo
print(f"CUDA is available? {torch.cuda.is_available()}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)



train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((256,256)),
    T.ToTensor(),
    #T.ColorJitter(contrast=2),
    #T.RandomHorizontalFlip(p=0.5)
])

transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((256,256)),
    T.ToTensor()
])

#divide il dataset
train_dataset = NeuronSegmentation(root_dir = data_dir, split='train', transforms = train_transforms)
val_dataset = NeuronSegmentation(root_dir = data_dir, split='val', transforms = transforms)
test_dataset = NeuronSegmentation(root_dir = data_dir, split='test', transforms = transforms)
# creazione loader 
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, num_workers=0, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, num_workers=0, shuffle=False, drop_last=True)

# Definiamo un dizionario dei loader
loaders = {"train": train_loader,
           "val": val_loader,
           "test": test_loader}



total_num_blacks = 0
total_num_whites = 0
 
for batch in train_loader:
    images, masks = batch['image'], batch['mask']
    target = masks
    num_blacks = (target == 0).sum().item()
    num_whites = (target == 1).sum().item()
    total_num_blacks += num_blacks
    total_num_whites += num_whites
 
print(f"Somma di Blacks in tutto il dataset: {total_num_blacks}")
print(f"Somma di Whites in tutto il dataset: {total_num_whites}")
 
print(f"neri su totali :{total_num_blacks/(total_num_blacks + total_num_whites)}")
 
balancedweight=(total_num_blacks/total_num_whites)/3 
#dalla matrice di confusione notiamo che ci sono molti neri predetti bianchi in proporzione ai bianchi predetti bianchi,
# dovuto al rumore quindi cerchiamo di bilanciare e facendo una proporzione diviadiamo per 3


# Pesi delle classi
weights = torch.tensor([1., balancedweight], device=dev)
print(weights)
# Definisci una loss function
criterion = nn.CrossEntropyLoss(weight=weights)

model = Model(dropout_prob=0.1)

#optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)


for split in ["train", "val", "test"]:
    print(f"Length of {split} loader: {len(loaders[split])}")

train( model, criterion, 18 ,loaders, dev, lr=0.0001, load_checkpoint = False, save_every = 1, save_path = 'weights')