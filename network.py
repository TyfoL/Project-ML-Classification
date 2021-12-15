import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
device = torch.device("cpu")

# By Xueyao JI ----------------------------------------
class array2dict(torch.utils.data.Dataset):

    def __init__(self, X_train, y_train):
        self.x_data = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.y_data = torch.tensor(y_train, dtype=torch.float32).to(device)
        self.y_data = self.y_data.reshape(-1,1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
          idx = idx.tolist()
        feature = self.x_data[idx,:]
        label   = self.y_data[idx,:]
        dict_tensor  = { 'feature' : feature, 'label' : label }
        return dict_tensor

# By Xueyao JI ----------------------------------------
def accuracy(model, data_tensor):
    n_correct, n_wrong = 0, 0
    for i in range(len(data_tensor)):
        feature = data_tensor[i]['feature']
        label   = data_tensor[i]['label']    # float32  [0.0] or [1.0]
        with torch.no_grad():
            output = model(feature)
        if label<0.5 and output<0.5:
            n_correct += 1
        elif label>=0.5 and output>=0.5:
            n_correct += 1
        else:
            n_wrong += 1
    return (n_correct*1.0)/(n_correct+n_wrong)

# By Xueyao JI ----------------------------------------
class NetSimple(torch.nn.Module):
    def __init__(self):
        super(NetSimple, self).__init__()
        self.hid1 = torch.nn.Linear(4, 8)
        self.hid2 = torch.nn.Linear(8, 8)
        self.oupt = torch.nn.Linear(8, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight) 
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight) 
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight) 
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        output = torch.tanh(self.hid1(x)) 
        output = torch.tanh(self.hid2(output))
        output = torch.sigmoid(self.oupt(output)) 
        return output

# By Xueyao JI ----------------------------------------
class NetDeeper(torch.nn.Module):
    def __init__(self):
        super(NetDeeper, self).__init__()
        self.hid1 = torch.nn.Linear(24, 36)
        self.hid2 = torch.nn.Linear(36, 20)
        self.hid3 = torch.nn.ReLU()
        self.hid4 = torch.nn.Linear(20, 8)
        self.oupt = torch.nn.Linear(8, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight) 
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight) 
        torch.nn.init.zeros_(self.hid2.bias)
        #torch.nn.init.xavier_uniform_(self.hid3.weight) 
        #torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.hid4.weight) 
        torch.nn.init.zeros_(self.hid4.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight) 
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        output = torch.tanh(self.hid1(x)) 
        output = torch.tanh(self.hid2(output))
        output = torch.tanh(self.hid3(output))
        output = torch.tanh(self.hid4(output))
        output = torch.sigmoid(self.oupt(output)) 
        return output

# By Xueyao JI ----------------------------------------
def network(data, net="simple", batch_size=10, n_epochs=100, lr=0.01):

    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    train_tensor = array2dict(X_train, y_train)
    test_tensor  = array2dict(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=True)
    train_losses, valid_losses = [], []

    # get model
    if net == "simple":
        net = NetSimple().to(device)
    elif net == "deeper":
        net = NetDeeper().to(device)
    net = net.train()
    criterion = torch.nn.BCELoss() # binary cross entropy
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # train
    for epoch in range(0, n_epochs):
        epoch_loss = 0.0
        valid_loss = 0.0

        for (index, batch) in enumerate(train_loader):
            X = batch['feature'] # [10,4]
            Y = batch['label']   # [10,1]
            output = net(X)      # [10,1]

            loss = criterion(output, Y) # train_y
            epoch_loss += loss.item()

            optimizer.zero_grad() # initialize gradients to zero
            loss.backward()   # compute gradients
            optimizer.step()      # update weights

        #train_losses.append(loss.detach().numpy())
        train_losses.append(epoch_loss)

        if epoch % 20 == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        # valid
        for (index, batch) in enumerate(test_loader):
            Xtest = batch['feature']
            Ytest = batch['label']
            with torch.no_grad():
                output_test = net(Xtest)
                loss_val = criterion(output_test, Ytest)
                valid_loss += loss_val.item()
        valid_losses.append(valid_loss)

    # plot
    plt.plot(range(n_epochs), train_losses)
    plt.plot(range(n_epochs), valid_losses)
    plt.legend(['train_losses', 'valid_losses'], prop={'size': 10})
    plt.title('loss curve', size=10)
    plt.xlabel('epoch', size=10)
    plt.ylabel('loss value', size=10)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig('loss_curve-{}.jpg'.format(current_time))

    # evaluate model
    net = net.eval()
    acc_train = accuracy(net, train_tensor)
    print("train accuracy = %0.2f%%" % (acc_train*100))
    acc_test = accuracy(net, test_tensor )
    print("test accuracy = %0.2f%%" % (acc_test*100))
    