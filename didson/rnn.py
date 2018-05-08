import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from didson.data import SequenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainpath = "labeled"
testpath = "labeled_test"

trainset = SequenceDataset(trainpath)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = SequenceDataset(testpath)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 64, 1, 1, 0)

    def forward(self, x, mask):
        x = torch.cat((x, mask), 0)
        x = x.unsqueeze(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        return x.squeeze(0)


class RNN(nn.Module):
    def __init__(self, hidden_size=15*64):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = Embedding()
        self.center_embed = nn.Linear(2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        frame, mask, center = input
        embedded = self.embedding(frame, mask).view(1, 1, -1)
        center = self.center_embed(center.view(1, -1)).view(1, 1, -1)
        output = embedded + center
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Predictor(nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = RNN(hidden_size)
        self.predictor = nn.Linear(hidden_size, 3)

    def forward(self, x):
        frame, mask, center = x
        frame = frame.squeeze(0)
        mask = mask.squeeze(0)
        center = center.squeeze(0)

        input_length = frame.size(0)
        encoder_hidden = self.rnn.initHidden()

        encoder_output = torch.zeros(self.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.rnn(
                (frame[ei], mask[ei], center[ei]), encoder_hidden)
        return self.predictor(encoder_output).squeeze(0)


predictor = Predictor(15*64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(predictor.parameters(), lr=0.0001)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        inputs, labels = data
        prediction = predictor(inputs)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

correct = 0
total = 0
counts = {
    0: {0: 0, 1: 0, 2: 0},
    1: {0: 0, 1: 0, 2: 0},
    2: {0: 0, 1: 0, 2: 0},
}
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = predictor(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        out = labels.numpy()[0]
        pred = predicted.numpy()[0]
        counts[out][pred] += 1

print('Accuracy of the network on the test sequences: %d %%' % (
    100 * correct / total))
print(counts)
