import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
  
# Inizializza il modello pre-trained Unet  da segmentation_models_pytorch
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
 
model_smp = smp.Unet(
   
    encoder_name="resnet50",            # Nome dell'encoder
    encoder_depth=5,                    # Profondità dell'encoder
    encoder_weights="imagenet",         # Pesi pre-addestrati dell'encoder
    decoder_use_batchnorm=True,         # Batch normalization nel decoder
    decoder_channels=[256, 128, 64, 32, 16],  # Canali del decoder
    decoder_attention_type=None,        # Tipo di attenzione nel decoder, se presente
    in_channels=1,                      # Numero di canali in ingresso (1 per immagini in scala di grigi)
    classes=2,                          # Numero di classi di output (1 per segmentazione binaria)
    activation='sigmoid',               # Funzione di attivazione
)

 

encoder = model_smp