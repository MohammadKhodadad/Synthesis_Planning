import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import random

# Function to read and parse the text file
def read_reaction_data(file_path):
    reactions = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                reactants, products = line.strip().split(">>")
                reactants=reactants.split('.')
                products=products.split('.')
                for react in reactants:
                    for prod in products:
                        reactions.append((react, prod))

            except:
                pass
    return reactions

# Custom Dataset class
class ReactionDataset(Dataset):
    def __init__(self, reactions):
        self.reactions = reactions

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        reactants, products = self.reactions[idx]
        return reactants, products

# Function to generate a negative sample
def generate_negative_sample(batch, exclude_idx):
    idx = random.choice(range(len(batch)))
    while idx == exclude_idx:
        idx = random.choice(range(len(batch)))
    return batch[idx]

# Custom collate function to generate negatives within a batch
def collate_fn(batch):
    positives = batch
    negatives = []

    batch_size = len(batch)
    for i in range(batch_size):
        reactants = batch[i][0]
        negative_sample = generate_negative_sample(batch, i)
        negatives.append((reactants, negative_sample[1]))  # Use the same reactants with a different product

    combined = positives + negatives
    labels = [1] * batch_size + [0] * batch_size
    
    reactants, products = zip(*combined)
    
    return reactants, products, torch.tensor(labels)


def load_data():
    file_path = 'data/reactionSmilesFigShare2023.txt'
    reaction_data = read_reaction_data(file_path)
    print(f'the number of records is: {len(reaction_data)}')
    reaction_dataset = ReactionDataset(reaction_data)
    data_loader = DataLoader(reaction_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    return data_loader
