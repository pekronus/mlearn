import numpy as np
import scipy as sp

from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 11, bias=False, dtype=torch.float64)
        nn.init.constant_(self.fc1.weight, 0.0)
    #    self.drop = nn.Dropout(0.5)
    #    self.fc2 = nn.Linear(11, 11, bias= False)

    def forward(self, xb):
        xb = self.fc1(xb)
     #   #xb = self.drop(xb)
     #   #xb = F.tanh(self.fc2(xb))
        return xb

def testf1(batch, lr = 0.01, lr_scale = 0.95, Nepoch = 10):
    print('Testf1')
    # create a model
    mdl = net1()


    a = batch[0]['x']
    y = batch[0]['y']

    losses = []
    for epoch in range(1, Nepoch):
        if (epoch % 20 == 0):
            lr *= lr_scale

        out = mdl(a)
        loss = F.mse_loss(out, y)
        losses.append(loss.item())
        print("Epoch: ", epoch, ' lr = ', lr, ' loss:', loss.item())

        # take the backward() for y
        loss.backward()
        # update parameters
        with torch.no_grad():
            for p in mdl.parameters():
                #print('p grad: ', p.grad)
                p -= p.grad * lr
                #print('p = ', p)
            mdl.zero_grad()

    print('-------------------')
    # loss = 0.5 * (A @ a - y).T @ (A @ a - y)
    out = mdl(a)
    loss = F.mse_loss(out, y)
    print('Final loss:', loss.item())
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()
    for p in mdl.parameters():
        print('params = ', p)


#    print('A@a - y: ', (A@a - y) )
# optimizer = torch.optim.SGD(A, lr=0.01)

def testf(batch, lr = 0.01, lr_scale = 0.95, Nepoch = 10):
    print('Testf')
    def mdl(x, A):
        return x@A.T


    #a = torch.tensor(np.array([1.0,1.0,1.0]).reshape(3,1), requires_grad= False)
    A = torch.tensor(np.zeros((11,3)), requires_grad = True)
    #y = torch.tensor(np.array([1+i + i*i for i in range(11)]).reshape(11,1), requires_grad = False)

    a = batch[0]['x']
    y = batch[0]['y']

    losses = []
    for epoch in range(1,Nepoch):
        if (epoch%20 == 0):
            lr *= lr_scale

        # define a function y
        # with torch.no_grad():
        #     t = A@a - y
        #     print('t = ', t, t.T@t)
        out = mdl(a,A)
        loss = F.mse_loss(out, y)
        losses.append(loss.item())
        #loss = (A@a - y).sum()
        print("Epoch: ", epoch, ' lr = ', lr, ', loss = ', loss.item())

        # take the backward() for y
        loss.backward()
        # print the gradients w.r.t. above x, w, and b
        #print("A.grad:", A.grad)
        # print("y.grad:", y.grad)
        # print("a.grad:", a.grad)

        with torch.no_grad():
            A -= A.grad*lr
            A.grad.zero_()
        print("A:", A)


    print('-------------------')
    #loss = 0.5 * (A @ a - y).T @ (A @ a - y)
    out = mdl(a, A)
    loss = F.mse_loss(out, y)
    print('Final loss:', loss.item())
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()
    print("A:", A)
#    print('A@a - y: ', (A@a - y) )
    #optimizer = torch.optim.SGD(A, lr=0.01)


def create_data_for_interp(f, nparams, nbatches=1, batch_size = 100):
    batches = []
    x = np.linspace(0, 10, 11)
    for i in range(nbatches):
        x_data = np.random.rand(batch_size, nparams)
        y_data = np.zeros([batch_size, x.size])
        for j in range(batch_size):
            y_data[j, :] = f(x, x_data[j, :])
        batches.append({
            'x': torch.tensor(x_data, dtype=torch.float64),
            'y': torch.tensor(y_data, dtype=torch.float64
        )})
    return batches

def create_data(nbatches=1, batch_size = 100, data_len = 100, right_pt = 10.0):
    batches = []
    for i in range(nbatches):
        x_data = right_pt * np.random.rand(data_len, batch_size)
        x_data = np.sort(x_data, axis = 0)
        y_data = vf2l(x_data)
        batches.append({
            'x': torch.tensor(x_data, dtype=torch.float64),
            'y': torch.tensor(y_data, dtype=torch.float64
        )})
    return batches

def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x and y
        x, y = batch['x'], batch['y']

        # Get output predictions
        out = model(x)

        # Compute loss
        loss = F.mse_loss(out, y)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            #print(model.fc1.weight.grad)
            optimizer.step()
            #print("weights =: ", model.fc1.weight)

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss


def func_to_learn(x, params):
    r = 0
    for i in range(params.size):
        r += params[i]*x**i
    return r


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    vf2l = np.vectorize(func_to_learn, excluded={1})
    #batch = create_data_for_interp(vf2l, 3, nbatches=1, batch_size=100)
    # #testf(batch, lr=0.01, lr_scale=0.95, Nepoch=1000)
    #testf1(batch, lr=0.01, lr_scale = 1, Nepoch=10000)
    #exit()

    train_data = create_data_for_interp(vf2l, 3,nbatches = 1, batch_size=100)
    model= net1()
    #model.fc1.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0, nesterov=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=False
    )
    losses = []
    for epoch in range(1, 2000):
        #print("-------------\nEpoch {}:\n".format(epoch))
        # Run **training***
        loss = run_epoch(train_data, model.train(), optimizer)
        losses.append(loss)
        print('Epoch: {} Train loss: {:.6f}, lr = {}'.format(epoch, loss, optimizer.param_groups[0]['lr']))        # Run **validation**
        #val_loss = run_epoch(val_data, model.eval(), optimizer)
        scheduler.step(loss)
    model.eval()

    for p in model.parameters():
        print(p)

    #plot losses
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()