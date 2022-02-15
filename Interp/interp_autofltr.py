import numpy as np
import scipy as sp

from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


#
# Here the goal is to replicate the input.  We will shrink the network to deduce the key functional
#
class net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(11, 3, bias=False, dtype=torch.float64)
        self.fc2 = nn.Linear(3, 11, bias=False, dtype=torch.float64)
        nn.init.constant_(self.fc1.weight, 0.0)
        nn.init.constant_(self.fc1.weight, 0.0)
    #    self.drop = nn.Dropout(0.5)
    #

    def forward(self, xb):
        xb = self.fc1(xb)
        xb = self.fc2(xb)
     #   #xb = self.drop(xb)
     #   #xb = F.tanh(self.fc2(xb))
        return xb


def create_data_for_interp(f, nparams, nbatches=1, batch_size = 100):
    batches = []
    xss = []
    x = np.linspace(0, 10, 11)
    for i in range(nbatches):
        xs = np.random.rand(batch_size, nparams)
        xss.append(xs)
        y_data = np.zeros([batch_size, x.size])
        for j in range(batch_size):
            y_data[j, :] = f(x, xs[j, :])

        x_data = y_data
        batches.append({
            'x': torch.tensor(x_data, dtype=torch.float64),
            'y': torch.tensor(y_data, dtype=torch.float64
        )})
    return batches, xss



def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, accuracy"""
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

def plot_results(val_data, xss):
    batch = val_data[0]
    x, y = batch['x'], batch['y'].numpy()
    out = model(x).detach().numpy()
    inner_layer = model.fc1(x).detach().numpy()
    print(y[1], out[1])
    print('xss = "', xss)
    print('inn layer = ', inner_layer)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(y[0], out[0])
    axs[1, 0].scatter(y[1], out[1])
    axs[0, 1].scatter(y[2], out[2])
    axs[1, 1].scatter(y[3], out[3])

    plt.show()


def func_to_learn(x, params):
    r = 0
    for i in range(params.size):
        r += params[i]*x**i
    #r = np.sin(x)*x
    return r


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    vf2l = np.vectorize(func_to_learn, excluded={1})

    train_data, txss = create_data_for_interp(vf2l, 3,nbatches = 20, batch_size=100)
    model= net1()
    #model.fc1.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0, nesterov=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=False
    )
    losses = []
    for epoch in range(1, 5000):
        #print("-------------\nEpoch {}:\n".format(epoch))
        # Run **training***
        loss = run_epoch(train_data, model.train(), optimizer)
        losses.append(loss)
        print('Epoch: {} Train loss: {:.6f}, lr = {}'.format(epoch, loss, optimizer.param_groups[0]['lr']))        # Run **validation**
        #val_loss = run_epoch(val_data, model.eval(), optimizer)
        scheduler.step(loss)

    model.eval()

    val_data, vxss = create_data_for_interp(vf2l, 3, nbatches=1, batch_size=10)
    loss = run_epoch(train_data, model.eval(), optimizer)

    print('Validation loss:', loss)
    plot_results(val_data, vxss)
    #plot_results(train_data, txss)

    for p in model.parameters():
        print(p)

    #plot losses
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()

