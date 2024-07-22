import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, jaccard_score
import seaborn as sns
import numpy as np
 
def train(model, criterion, epochs, loaders, dev, lr=0.0001, load_checkpoint=False, save_every=10, save_path='weights'):
    try:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # sposta  model al device
        model = model.to(dev)
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if load_checkpoint:
            if os.path.isfile(os.path.join(save_path, 'weights.pt')):
                print('Loading weights...')
                model.load_state_dict(torch.load(os.path.join(save_path, 'weights.pt')))
            if os.path.isfile(os.path.join(save_path, 'optim.pt')):
                print('Loading optimizer...')
                optimizer.load_state_dict(torch.load(os.path.join(save_path, 'optim.pt')))
            print('Loading completed!')
        # Inizializza l'history
        history_loss = {"train": [], "val": [], "test": []}
        history_accuracy = {"train": [], "val": [], "test": []}
        history_iou = {"train": [], "val": [], "test": []}
        # Processa ogni epoca
        for epoch in range(epochs):
            # Initialize le variabili delle epoche 
            sum_loss = {"train": 0, "val": 0, "test": 0}
            sum_accuracy = {"train": 0, "val": 0, "test": 0}
            sum_iou = {"train": 0, "val": 0, "test": 0}
            count = {"train": 0, "val": 0, "test": 0}
            # Processa ogni split
            for split in ["train", "val", "test"]:
                #Imposta il modello in modalita training o valutazione
                torch.set_grad_enabled(split == 'train')
                model.train() if split == 'train' else model.eval()
                # Inizializza la matrice di confusione 
                all_preds = []
                all_targets = []
                # Processa ogni batch
                for batch in loaders[split]:
                    inputs = batch['image'].to(dev)
                    targets = batch['mask'].squeeze(1).to(dev)
                    # Resetta i gradienti
                    optimizer.zero_grad()
                    # Calcola l'output
                    output = model(inputs)
                    loss = criterion(output, targets.long())
                    # Aggiorna loss
                    sum_loss[split] += loss.item()
                    if split == "train":
                        # Calcola i gradienti e ottimizza
                        loss.backward()
                        optimizer.step()
                    # Calcola accuracy e IoU
                    pred = torch.argmax(output, 1)
                    batch_accuracy = (pred == targets).sum().item() / targets.numel()
                    sum_accuracy[split] += batch_accuracy
                    count[split] += 1
 
                    all_preds.append(pred.cpu().numpy().flatten())
                    all_targets.append(targets.cpu().numpy().flatten())
                # Calcola matrice di confusione  e IoU
                all_preds = np.concatenate(all_preds)
                all_targets = np.concatenate(all_targets)
                # Assicura il formato binario per il calcolo della matrice di confusione  e dell'IoU
                all_preds_binary = (all_preds > 0.5).astype(int)
                all_targets_binary = (all_targets > 0.5).astype(int)
                cm = confusion_matrix(all_targets_binary, all_preds_binary)
                iou = jaccard_score(all_targets_binary, all_preds_binary, average='macro')
                sum_iou[split] = iou
 
                # Salva  checkpoint del modello
                if epoch % save_every == 0 and split == 'train':
                    torch.save(model.state_dict(), os.path.join(save_path, 'weights.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(save_path, 'optim.pt'))
            # Calcola epoch loss, accuracy, and IoU
            epoch_loss = {split: sum_loss[split] / count[split] for split in ["train", "val", "test"]}
            epoch_accuracy = {split: sum_accuracy[split] / count[split] for split in ["train", "val", "test"]}
            epoch_iou = {split: sum_iou[split] for split in ["train", "val", "test"]}
            # Aggiorna l'history
            for split in ["train", "val", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
                history_iou[split].append(epoch_iou[split])
            # stampa tutte le metriche dell'epoca
            print(f"Epoch {epoch+1}:",
                  f"TrL={epoch_loss['train']:.4f},",
                  f"TrA={epoch_accuracy['train']:.4f},",
                  f"TrIoU={epoch_iou['train']:.4f},",
                  f"VL={epoch_loss['val']:.4f},",
                  f"VA={epoch_accuracy['val']:.4f},",
                  f"VIoU={epoch_iou['val']:.4f},",
                  f"TeL={epoch_loss['test']:.4f},",
                  f"TeA={epoch_accuracy['test']:.4f},",
                  f"TeIoU={epoch_iou['test']:.4f},")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.title("Loss")
        for split in ["train", "val", "test"]:
            plt.plot(history_loss[split], label=split)
        plt.legend()
        plt.show()
        # Plot accuracy
        plt.figure(figsize=(12, 6))
        plt.title("Accuracy")
        for split in ["train", "val", "test"]:
            plt.plot(history_accuracy[split], label=split)
        plt.legend()
        plt.show()
 
        # Plot IoU
        plt.figure(figsize=(12, 6))
        plt.title("IoU")
        for split in ["train", "val", "test"]:
            plt.plot(history_iou[split], label=split)
        plt.legend()
        plt.show()
        # Plot confusion matrix for the test split
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_targets_binary, all_preds_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Black', 'White'], yticklabels=['Black', 'White'])
        plt.title("Confusion Matrix (Test Split)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()