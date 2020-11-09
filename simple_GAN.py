import torch
import torch.nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#performance tips
# larger network?
# better normalization with BatchNorm
# Different learning rates
# change architecture to CNN

class Discriminator(nn.Module):
    
    def __init__(self, img_dim): #from MNIST dataset so images 28x28x1 --> 784
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1), #slope can be another HP
            nn.Linear(128, 1), #single output because real or fake
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    
    def __init__(self, z_dim, img_dim): #z_dim is the dimension of the latent noise
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1), #leakyReLU is the standard default for GANs
            nn.Linear(256, img_dim), #producing fake image of same dimension,
            nn.Tanh(), #normalizing values to be between -1 and 1, so that it corresponds to the same normalization of our pixel data
        )

    def forward(self, x):
        return self.gen(x)

#HPs
# Simple GANs are very sensitive to HPs 
# Newer GANs are more robust

#if not using in cluster
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim = 64 #128, 256
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

#setup
disc = Discriminator(image_dim)#.to(device)
gen = Generator(z_dim, image_dim)#.to(device)
fixed_noise = torch.randn((batch_size, z_dim))#.to(device) #to visualize how it changes across epochs on tensorboard 
transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))] 
)
dataset = datasets.MNIST(root = 'dataset/', transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss() # almost the same form discussed in the introduction
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake') #for tensorboard
writer_real = SummaryWriter(f'runs/GAN_MNIST/real') 
step = 0 #also for tensorboard

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784)#.to(device) # -1 to keep number of examples in batch, 784 to flatten the images
        batch_size = real.shape[0]

        ### Training Discriminator: max log(D(real)) + log(1 - D(G(x)) OR equiv to min -XX
        noise = torch.randn(batch_size, z_dim)#.to(device)
        fake = gen(noise) #generating a fake image
        disc_real = disc(real).view(-1) #first part log(D(real))
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1) 
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True) #read explanation below
        opt_disc.step()

        ### Train Generator min log(1 - D(G(z))) --> saturated gradients
        ## so instead max log(D(G(z))), which is the same that we did with fake = gen(noise)
        ## since we want to reuse that info we don't let .backward to clear the cached info

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1