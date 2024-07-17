import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import random




# Check for available CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the ChemBERTa Embedder
class ChemBERTaEmbedder(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", embedding_dim=128):
        super(ChemBERTaEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.chemberta.config.hidden_size, embedding_dim)
        
        # Freeze the ChemBERTa parameters
        for param in self.chemberta.parameters():
            param.requires_grad = False
    
    def forward(self, smiles):
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU if available
        outputs = self.chemberta(**inputs)
        pooled_output = self.pooler(outputs.last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        embeddings = self.linear(pooled_output)
        return embeddings

# Define the Custom Loss
class CustomLoss(nn.Module):
    def __init__(self, margin):
        super(CustomLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred, intersect_embs, labels):
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0, device=emb_as.device), margin - e)[labels == 0]

        relation_loss = torch.sum(e)
        return relation_loss


def load_model(embedding_dim = 128):
    model = ChemBERTaEmbedder(embedding_dim=embedding_dim).to(device)
    return model


def train(model, data_loader):
    
    margin = 1.0
    learning_rate = 1e-4
    num_epochs = 10

    # Instantiate model and loss function
    criterion = CustomLoss(margin=margin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm.tqdm(data_loader):
            reactants, products, labels = batch
            
            # Move labels to GPU if available
            labels = labels.to(device)
            
            # Convert SMILES strings to embeddings
            emb_as = model(list(reactants))
            emb_bs = model(list(products))
            
            # Compute the loss
            loss = criterion((emb_as, emb_bs), None, labels)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), 'weights/model.pth')