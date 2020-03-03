# Zhu Jianing
# Pytorch practice: style_transformation & GAN


from __future__ import division
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np 


import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
transformss = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))

])
mnist_data = datasets.MNIST("./mnist_data", train = True, download=True, transform=transformss)
dataloader = torch.utils.data.DataLoader( dataset=mnist_data, batch_size = batch_size, shuffle=True)

plt.imshow(next(iter(dataloader))[0][0][0])
image_size = 28 * 28
hidden_size = 256
latent_size = 64
# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)


# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)
D = D.to(device)
G = G.to(device)

loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0002)


total_steps = len(dataloader)
num_epochs = 30
for epochs in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.shape[0]
        images = images.reshape(batch_size, image_size).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(images)
        d_loss_real = loss(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = loss(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        outputs = D(fake_images)
        g_loss = loss(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 200 == 0 :
            print("Epoch:{},Step:{},d_loss:{:.4f},g_loss:{:.4f}".format(epochs,i,d_loss.item(),g_loss.item()))