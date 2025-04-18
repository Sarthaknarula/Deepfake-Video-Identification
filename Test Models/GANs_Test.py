import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of the image being real
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
z_dim = 100  # Latent space size for the generator
image_dim = 28 * 28  # Image size (28x28 for MNIST)
num_epochs = 100

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to range [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models, optimizers, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(input_dim=z_dim, output_dim=image_dim).to(device)
discriminator = Discriminator(input_dim=image_dim).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.view(-1, 28 * 28).to(device)
        batch_size = real_images.size(0)

        # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Real images
        real_outputs = discriminator(real_images)
        d_loss_real = loss_fn(real_outputs, real_labels)

        # Fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        d_loss_fake = loss_fn(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator: maximize log(D(G(z)))
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = loss_fn(outputs, real_labels)  # Trick discriminator

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

# Generate and visualize some images
z = torch.randn(batch_size, z_dim).to(device)
generated_images = generator(z).view(-1, 28, 28).cpu().detach()

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(generated_images[i], cmap='gray')
    axes[i].axis('off')
plt.show()
