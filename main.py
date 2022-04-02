import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import *
import numpy as np
from torch.utils.data import Dataset
import csv

batch_size = 256
torch.multiprocessing.set_sharing_strategy('file_system')


class TextDataset(Dataset):
    def __init__(self, test, text, text2, label=None):
        self.test = test
        self.text2 = text2
        self.text = text
        self.label = label
        self.batch_tokenized = tokenizer.batch_encode_plus(text, add_special_tokens=True,
                                                           max_length=64, truncation=True, pad_to_max_length=True)
        self.batch_tokenized2 = tokenizer.batch_encode_plus(text2, add_special_tokens=True,
                                                            max_length=64, truncation=True, pad_to_max_length=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = (torch.tensor(self.batch_tokenized["input_ids"][item]),
                torch.tensor(self.batch_tokenized["attention_mask"][item]),
                torch.tensor(self.batch_tokenized2["input_ids"][item]),
                torch.tensor(self.batch_tokenized2["attention_mask"][item]))
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
    #for i in range(n):
    #    a = random.random()
    #    x = random.randint(0, n - 1)
    #    if a < 0.5:
    #        train_text.append(train_text[x])
    #    else:
    #        train_text.append(train_text2[x])
    #    a = random.random()
    #    y = random.randint(0, n - 1)
    #    while x == y:
    #        y = random.randint(0, n - 1)
    #    if a < 0.5:
    #        train_text2.append(train_text[y])
    #    else:
    #        train_text2.append(train_text2[y])
    #    label.append(0)
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
    model = Bert_model(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Start training!")

    # train
    batch_count = len(train_text) // batch_size
    model.train()
    for epoch in range(20):
        print_avg_loss = 0
        for batch_idx, ((x, mx, x2, mx2), label) in enumerate(train_loader):
            x = x.to(device)
            mx = mx.to(device)
            x2 = x2.to(device)
            mx2 = mx2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x, mx, x2, mx2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
        print("Epoch: %d, Loss: %.4f" % ((epoch + 1), print_avg_loss / batch_count))

    # test
    model.eval()
    preds = np.array([])
    with torch.no_grad():
        for batch_idx, ((x, mx, x2, mx2), _) in enumerate(test_loader):
            x = x.to(device)
            mx = mx.to(device)
            x2 = x2.to(device)
            mx2 = mx2.to(device)
            output, _ = model(x, mx, x2, mx2)
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
