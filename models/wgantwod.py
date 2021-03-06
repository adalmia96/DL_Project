import os, sys
sys.path.append(os.getcwd())

import time
import functools
#import argparse

import numpy as np
#import sklearn.datasets

#from models.wgan import *

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
#import torch.nn.init as init
import pickle
import preprocessing as pp
import nltk
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import math

# Replace your training data path here
DATA_DIR = './cache/training/'
# Output path where result will be stored
OUTPUT_PATH = './output/'

# if True, it will load saved model from OUT_PATH and continue to train
#RESTORE_MODE = False
# Starting iteration
#START_ITER = 0
# Model dimensionality
#DIM = 50
# How many iterations to train the critic for
#CRITIC_ITERS = 2
# How many iterations to train the generator for
#GENER_ITERS = 1
# Number of GPUs
#N_GPUS = 1
# Batch size. Must be a multiple of N_GPUS
#BATCH_SIZE = 64
# How many iterations to train for
#END_ITER = 2
# Gradient penalty lambda hyperparameter
#LAMBDA = 10
# Number of pixels in each word matrix (word vector length * sequence length)
#OUTPUT_DIM = 50*50*1

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                nn.init.kaiming_uniform_(m.conv.weight)
            else:
                nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# Create the dataset
def loader_funct(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def calc_gradient_penalty(netD, real_data, fake_data, lambda_term, batch_size, dim1, dim2):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 1, dim1, dim2)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 1, dim1, dim2)
    real_data = real_data.view(batch_size, 1, dim1, dim2)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return gradient_penalty

def generate_image(netG, batch_size, noise=None, dim1=50, dim2=50):
    if noise is None:
        noise = gen_rand_noise(batch_size)

    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    samples = samples.view(batch_size, 1, dim1, dim2)
    #samples = samples * 0.5 + 0.5
    return samples

def gen_rand_noise(batch_size):
    noise = torch.randn(batch_size, 128)
    noise = noise.to(device)

    return noise


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, height, width, resample=None):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, height, width])
            self.bn2 = nn.LayerNorm([input_dim, height, width])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, height, width])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output

class Generator(nn.Module):
    def __init__(self, dim1, dim2, output_dim):
        super(Generator, self).__init__()

        self.dim = dim1
        self.height = dim1
        self.width = dim2
        self.output_dim = output_dim
        self.height_mult50 = int(self.height/50) # Multiple of 50
        self.width_mult50 = int(self.width/50) # Multiple of 50
        ks_sp = (self.height - self.height_mult50 * 48 + 1, self.width - self.width_mult50 * 48 + 1) #calculation

        self.ln1 = nn.Linear(128, self.height_mult50*self.width_mult50*3*3*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, height=self.height, width=self.width, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, height=self.height, width=self.width, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, height=self.height, width=self.width, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, height=self.height, width=self.width, resample = 'up')
        self.conv_sp = nn.ConvTranspose2d(1*self.dim, 1*self.dim, ks_sp, stride=1, padding=0, bias = False) #added
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 1, 3)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()

    def forward(self, input):
        #breakpoint()
        output = self.ln1(input.contiguous()) #64 x 28800
        output = output.view(-1, 8*self.dim, 3*self.height_mult50, 3*self.width_mult50) #64 x 800 x 6 x 6
        #print('64 x 800 x 6 x 6')
        #print(output.size())
        output = self.rb1(output) 
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output) #64 x 100 x 96 x 96
        #print('64 x 100 x 96 x 96')
        #print(output.size())
        output = self.conv_sp(output) #64 x 100 x 100 x 100
        #print('64 x 100 x 100 x100')
        #print(output.size())

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        #output = self.tanh(output)
        output = output.view(-1, self.output_dim)
        return output

class Discriminator(nn.Module): 
    def __init__(self, dim1, dim2):
        super(Discriminator, self).__init__()
        self.dim = dim1
        self.height = dim1
        self.width = dim2
        self.height_mult50 = int(self.height/50) # Multiple of 50
        self.width_mult50 = int(self.width/50) # Multiple of 50
        ks_sp = (self.height - self.height_mult50 * 48 + 1, self.width - self.width_mult50 * 48 + 1) #calculation
        scaled_height = self.height_mult50 * 48
        scaled_width = self.width_mult50 * 48
        
        self.tanh = nn.Tanh() #added
        self.conv1 = MyConvo2d(1, self.dim, 3, he_init = False)
        self.conv_sp = nn.Conv2d(1*self.dim, 1*self.dim, ks_sp, stride=1, padding=0, bias = False) #added
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', height=scaled_height, width=scaled_width)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', height=int(scaled_height/2), width=int(scaled_width/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', height=int(scaled_height/4), width=int(scaled_width/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', height=int(scaled_height/8), width=int(scaled_width/8))
        self.ln1 = nn.Linear(self.height_mult50*self.width_mult50*4*4*8*self.dim, 1)

    def forward(self, input):
        #breakpoint()
        output = input.contiguous()
        output = output.view(-1, 1, self.height, self.width)
        output = self.tanh(output) #added
        output = self.conv1(output)
        output = self.conv_sp(output) #added

        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output) #64 x 800 x 6 x 6
        #print('64 x 800 x 6 x 6')
        #print(output.size())
        output = output.view(-1, self.height_mult50*self.width_mult50*4*4*8*self.dim) #144 x 12800
        output = self.ln1(output) #144 x 1
        #print('144 x 1')
        #print(output.size())
        output = output.view(-1) #144
        return output

def train(we_model, batch_size=64, epochs=10000, d_iters=5, g_iters=1, lambda_term=10, g_lr=0.0001, d_lr=0.0001, wv_length=50, \
seq_length=50, restore=False, patience=200, discriminator_file="discriminator.pt", generator_file="generator.pt"):

    print("Training!")
    #---------------------Initialize Stuff------------------------
    training_dataset = datasets.DatasetFolder(root=DATA_DIR, loader=loader_funct, extensions=".pickle",
                                                 transform=transforms.Compose([
                                                    torch.from_numpy])
                                                )
    # Create the dataloader
    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)

    output_dim = seq_length*wv_length*1
    if restore:
        aG = torch.load(OUTPUT_PATH + "generator.pt")
        aD = torch.load(OUTPUT_PATH + "discriminator.pt")
    else:
        aG = Generator(seq_length, wv_length, output_dim)
        aD = Discriminator(seq_length, wv_length)

        aG.apply(weights_init)
        aD.apply(weights_init)


    optimizer_g = torch.optim.Adam(aG.parameters(), lr=g_lr, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=d_lr, betas=(0,0.9))
    one = torch.tensor(1.0)
    mone = one * -1
    aG = aG.to(device)
    aD = aD.to(device)
    one = one.to(device)
    mone = mone.to(device)

    fixed_noise = gen_rand_noise(batch_size) # batch_size x 128

    #---------------------Start Actual Training------------------------
    dataloader = training_data_loader
    dataiter = iter(dataloader)
    disc_costs = np.zeros(epochs)
    gen_costs = np.zeros(epochs)
    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch: " + str(epoch))
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(g_iters):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise(batch_size) # batch_size x 128
            noise.requires_grad_(True)
            fake_data = aG(noise) #64 x 10000 
            gen_cost = aD(fake_data) #36
            gen_cost = gen_cost.mean() #1

            gen_cost.backward(mone)
            gen_cost = -gen_cost
            gen_costs[epoch] = gen_cost

        optimizer_g.step()
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(d_iters):
            print("Critic iter: " + str(i))

            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise(batch_size)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach() #64 x 10000
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, lambda_term, batch_size, seq_length, wv_length)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_costs[epoch] = disc_cost
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()

        #---------------VISUALIZATION---------------------
        #if True:
        if epoch % 100 == 99:
            gen_images = generate_image(aG, batch_size, noise=fixed_noise, dim1=seq_length, dim2=wv_length)
            sentences = ""
            for gen_image in gen_images:
                b = gen_image.detach().cpu().numpy()
                sentences = sentences + pp.decode_word_array(b, we_model) + "\n"
            with open(OUTPUT_PATH + 'samples_{}.txt'.format(epoch), 'w') as f:
                f.write(sentences)
            #writer.add_text('sentences', sentences, iteration)
	#----------------------Save model----------------------
            print("Saving models!")
            pickle.dump(gen_costs, open(OUTPUT_PATH + "gen_costs.p", "wb"))
            pickle.dump(disc_costs, open(OUTPUT_PATH + "disc_costs.p", "wb"))
            torch.save(aG, OUTPUT_PATH + generator_file)
            torch.save(aD, OUTPUT_PATH + discriminator_file)

def get_bert_score(sentence, tokenizer, bertMaskedLM):
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions=bertMaskedLM(tensor_input)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data
        return math.exp(loss)

def test(we_model, num_images=128, wv_length=50, seq_length=50, generator_file="generator.pt"):

    aG = torch.load(OUTPUT_PATH + generator_file)
    #aG.eval() #TODO check if this is necessary
    sentences = []
    images_per_batch = 32
    for i in range(int(num_images / images_per_batch)):
        gen_images = generate_image(aG, images_per_batch, dim1=seq_length, dim2=wv_length)
        for gen_image in gen_images:
            b = gen_image.detach().cpu().numpy()
            decode = pp.decode_word_array3(b, we_model)
            #print(decode)
            sentences.append(decode)

    # take first sentence and create an image
    #pp.create_fancy_image(' '.join(sentences[0]), we_model, seq_length, wv_length)

    bsss = []
    for t in sentences:
        bss = []
        for t2 in sentences:
            bss.append(nltk.translate.bleu_score.sentence_bleu([t2], t))
        print(np.average(bss))
        bsss.append(np.average(bss))
    print("Average BLEU Score", np.average(bsss))

    for i, s in enumerate(sentences):
        sentences[i] = " ".join(s)

    # BERT perplexity
    bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # make results deterministic
    bertMaskedLM.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Average BERT Perplexity", sum([get_bert_score(s, tokenizer, bertMaskedLM) for s in sentences]) / len(sentences))
