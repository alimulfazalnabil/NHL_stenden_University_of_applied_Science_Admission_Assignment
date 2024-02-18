import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        output = self.model(x)
        return output

# Define the Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        return output

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Discriminator and Generator networks
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Set up MNIST dataset and dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        real_outputs = discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        real_loss = criterion(real_outputs, real_targets)

        z = torch.randn(real_images.size(0), 100, device=device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        fake_targets = torch.zeros(real_images.size(0), 1, device=device)
        fake_loss = criterion(fake_outputs, fake_targets)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(real_images.size(0), 100, device=device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        targets = torch.ones(real_images.size(0), 1, device=device)
        g_loss = criterion(outputs, targets)

        g_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                  f'D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}')

# Save trained Generator model
torch.save(generator.state_dict(), 'mnist_generator.pth')
