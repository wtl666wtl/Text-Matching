import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from naive_model import *
import numpy as np
from torch.utils.data import Dataset
import csv
import torchtext.vocab as vocab
from util import *

cache_dir = '.vector_cache/glove'
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

batch_size = 256
torch.multiprocessing.set_sharing_strategy('file_system')

L = 32

class TextDataset(Dataset):
    def __init__(self, test, text, text2, label=None):
        self.test = test
        self.label = label
        self.text = text
        self.text2 = text2

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        t1 = tokenizer(self.text[item])
        t2 = tokenizer(self.text2[item])

        if len(t1) > L:
            t1 = t1[:L]
        if len(t2) > L:
            t2 = t2[:L]

        a = [[] for _ in range(L)]
        for i in range(L):
            for j in range(L):
                w1 = glove.vectors[glove.stoi["unk"]]
                w2 = glove.vectors[glove.stoi["unk"]]
                if i < len(t1) and t1[i] in glove.stoi:
                    w1 = glove.vectors[glove.stoi[t1[i].lower()]]
                if j < len(t2) and t2[j] in glove.stoi:
                    w2 = glove.vectors[glove.stoi[t2[j].lower()]]
                a[i].append(torch.sum(w1 * w2).item())

        text = torch.Tensor(a)
        if not self.test:
            return text, self.label[item]
        else:
            return text, []


def main():
    # input
    reader = csv.reader(open("train.tsv", "r", encoding="utf-8"), delimiter='\t')
    train_text = []
    train_text2 = []
    test_text = []
    test_text2 = []
    label = []
    for id, t1, t2, l in reader:
        if not id.isalpha():
            train_text.append(t1)
            train_text2.append(t2)
            label.append(int(l))
            train_text.append(t2)
            train_text2.append(t1)
            label.append(int(l))
    reader = csv.reader(open("test.tsv", "r", encoding="utf-8"), delimiter='\t')
    for id, t1, t2 in reader:
        if not id.isalpha():
            test_text.append(t1)
            test_text2.append(t2)

    test_set = TextDataset(test=True, text=test_text, text2=test_text2)
    train_set = TextDataset(test=False, text=train_text, text2=train_text2, label=label)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)
    device = torch.device('cuda')
    model = naive_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Start training!")

    # train
    batch_count = len(train_text) // batch_size
    model.train()
    for epoch in range(10):
        print_avg_loss = 0
        for batch_idx, (x, label) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
            print("OK")
        print("Epoch: %d, Loss: %.4f" % ((epoch + 1), print_avg_loss / batch_count))

    # test
    model.eval()
    preds = np.array([])
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(device)
            output = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
    preds = preds.astype(int)

    # output
    writer = csv.writer(open("submission.csv", "w", encoding="utf-8"))
    writer.writerow(("Id", "Category"))
    for i in range(len(preds)):
        writer.writerow((i, preds[i]))


if __name__ == '__main__':
    main()
