from utils.dataloader import *
from utils.model import *


data_loader=load_data()
model=load_model()
train(model,data_loader)