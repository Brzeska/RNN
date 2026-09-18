import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


train = True
print_loss = True
sample = False
print()

#read in data
words = open('train.txt','r').read().splitlines()
dev = open('dev.txt','r').read().splitlines()

#build mappings
chars = sorted(set('abcdefghijklmnopqrstuvwxyz'))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {stoi[s]:s for s in stoi}
vocab_size = len(itos)
#print(itos)
#print(vocab_size)


#build dataset
block_size = 3

X = []
Y = []

for w in words:
    w += '.'
    context = [0]*block_size
    for i in w:
        X.append(context)
        Y.append(stoi[i])
        context = context[1:] + [stoi[i]]

X = torch.tensor(X)
Y = torch.tensor(Y)

if train:
    #Build MLP
    n_embd = 10 #dimensionality of the character embedding matrix
    n_hidden = 200 #number of hidden neurons
    
    #note: kaiming init for tanh is (5/3)/sqrt(fan_in)

    C = torch.randn((vocab_size,n_embd)) #embedding matrix
    W1 = torch.randn((n_embd*block_size,n_hidden))*(5/3)/(block_size*n_embd)**0.5 #tanh layer
    b1 = torch.randn(n_hidden)*.01
    W2 = torch.randn((n_hidden,vocab_size))*.01 #softmax layer
    b2 = torch.randn(vocab_size) * 0

    bn_gain = torch.ones((1,n_hidden))
    bn_bias = torch.zeros((1,n_hidden))
    parameters = [C,W1,b1,W2,b2,bn_gain,bn_bias]
    
    #print(sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True


#define hyperparameters and train
max_steps = 200000
batch_size = 32
lossi = []
    
for i in range(max_steps):
    if not train:
        break
    #minibatch construct
    ix = torch.randint(0, X.shape[0],(batch_size,))
    Xb, Yb = X[ix], Y[ix]

    #forward pass
    emb = C[Xb]
    emb_cat = emb.view(emb.shape[0],-1)
    hpreact = emb_cat @ W1 + b1
    
    #apply batch normalization to the preactivations
    #this formula is taken from Ioffe/Szegedy
    #you also need to scale and shift
    hpreact = bn_gain*(hpreact - hpreact.mean(0, keepdim=True))/(hpreact.var(0, keepdim=True)+.001)**0.5 + bn_bias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    #backward pass
    for p in parameters:
        p.grad = None #zero gradients
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad

    #track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    
    #plt.hist(h.view(-1).tolist(),50)
    #plt.show()

if train:
    plt.plot(lossi)
    #plt.show()

if train:
    torch.save({
        'C': C,
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
    }, 'MLP.pt')


#get train and validation loss
model = torch.load('MLP.pt')
W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']
C = model['C']

#@torch.no_grad() #since we're not training, stop tracking gradients
with torch.no_grad():
    i=1
    emb = C[X]
    emb_cat = emb.view(emb.shape[0],-1)
    hpreact = emb_cat @ W1 + b1
    hpreact = bn_gain*(hpreact - hpreact.mean(0, keepdim=True))/(hpreact.var(0, keepdim=True)+.001)**0.5 + bn_bias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    if print_loss:
        print(f'train loss: {loss}')
    
    #Check for dead neurons
    #plt.figure()
    #plt.imshow(h.abs()>0.99, cmap='gray', interpolation='nearest')
    #plt.show()

    #build dev dataset
    X = []
    Y = []
    for w in dev:
        w += '.'
        context = [0]*block_size
        for i in w:
            X.append(context)
            Y.append(stoi[i])
            context = context[1:] + [stoi[i]]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    emb = C[X]
    emb_cat = emb.view(emb.shape[0],-1)
    hpreact = emb_cat @ W1 + b1
    hpreact = bn_gain*(hpreact - hpreact.mean(0, keepdim=True))/(hpreact.var(0, keepdim=True)+.001)**0.5 + bn_bias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    
    if print_loss: #really inefficient! Shouldn't compute if false
        print(f'dev loss: {loss}')
    
    #sample from model
    
    for _ in range(20):
        if not sample:
            break
        out = []
        context = [0]*block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1,-1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1).item()

            context = context[1:] + [ix]

            out.append(ix)

            if ix==0:
                break
            
        print(''.join(itos[i] for i in out))
