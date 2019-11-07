import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import helper


# Set random seed for reproducibility
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Size using in transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 500
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Create the dataset
dataset = dset.ImageFolder(root="./data",
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=9,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Random input distribution
fixed_noise = torch.randn(9, nz, 1, 1, device=device)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# Generator network
class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator network
class Discriminator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        )

    def forward(self, input):
        return self.main(input)


# Output 3x3 images from generator
def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()


# Create the generator
netG = Generator(ngpu).to(device)
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = torch.nn.DataParallel(netG, list(range(ngpu)))
    netD = torch.nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

# Print the model
print(netG)
print(netD)

# WGAN optimizers used RMSProp
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=5e-5)
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=5e-5)

# Training Loop
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # Back propagation parameter
    one = torch.FloatTensor([1]).to(device)
    mone = -1 * one
    # Output in print()
    errD_pr = 0
    errG_pr = 0
    D_x_pr = 0
    D_G_z1_pr = 0
    D_G_z2_pr = 0
    # For each batch in the dataloader
    with tqdm(total=len(dataloader)) as pbar:
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(output)
            errD_real.backward(one)
            D_x_pr = -output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(output)
            # Calculate the gradients for this batch
            errD_fake.backward(mone)
            # Update D
            optimizerD.step()
            errD = errD_fake - errD_real
            errD_pr = errD.item()
            D_G_z1_pr = output.mean().item()

            # Discriminator Clipper in WGAN
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = torch.mean(output)
            # Calculate gradients for G
            errG.backward(one)
            errG_pr = errG.item()
            D_G_z2_pr = output.mean().item()
            # Update G
            optimizerG.step()

            pbar.update()

    # Output training stats
    print('[%d/%d]\tLoss_D: %.4f\tWasserstein_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
          % (epoch+1, num_epochs,
             errD_pr, -errD_pr, errG_pr, D_x_pr, D_G_z1_pr, D_G_z2_pr))

    # Save the result in each epoch:
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    fake_batch = np.transpose(vutils.make_grid(fake, nrow=3, padding=0, normalize=True).numpy(), (1, 2, 0))
    generate_images = np.transpose(np.reshape(fake_batch, (3, 64, 3, 64, 3)), (0, 2, 1, 3, 4))
    generate_images = np.reshape(generate_images, (9, 64, 64, 3))
    output_fig(generate_images, file_name="images/{}_image".format(str.zfill(str(epoch+1), 3)))

# Save the model
torch.save(netD.state_dict(), './netD.pt')
torch.save(netG.state_dict(), './netG.pt')