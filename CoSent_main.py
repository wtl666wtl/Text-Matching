import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from CoSent import *
import numpy as np
from torch.utils.data import Dataset
import csv
from transformers import get_linear_schedule_with_warmup

Epoch = 5
batch_size = 32
torch.multiprocessing.set_sharing_strategy('file_system')


class TextDataset(Dataset):
    def __init__(self, test, text, text2, label=None):
        self.test = test
        self.label = label
        self.text = text
        self.text2 = text2

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        t1 = self.text[item]
        t2 = self.text2[item]
        encode1 = tokenizer.encode_plus(t1, add_special_tokens=True,
                                        max_length=256, truncation=True,
                                        pad_to_max_length=True, return_tensors='pt')
        encode2 = tokenizer.encode_plus(t2, add_special_tokens=True,
                                       max_length=256, truncation=True,
                                       pad_to_max_length=True, return_tensors='pt')
        text1 = (encode1['input_ids'].squeeze(0), encode1['attention_mask'].squeeze(0),
                encode1['token_type_ids'].squeeze(0))
        text2 = (encode2['input_ids'].squeeze(0), encode2['attention_mask'].squeeze(0),
                encode2['token_type_ids'].squeeze(0))
        if not self.test:
            return text1, text2, self.label[item]
        else:
            return text1, text2, []


def compute_sim(y_pred1, y_pred2):
    norms = (y_pred1 ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred1 = y_pred1 / norms
    norms = (y_pred2 ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_pred2 = y_pred2 / norms

    sim = torch.sum(y_pred1 * y_pred2, dim=1)
    return sim


def compute_loss(y_true, y_sim):
    y_sim = y_sim * 20
    y_sim = y_sim[:, None] - y_sim[None, :]
    y_true = y_true[:, None] < y_true[None, :]
    y_true = y_true.float()
    y_sim = y_sim - (1 - y_true) * 1e12
    y_sim = y_sim.view(-1)
    y_sim = torch.cat((torch.tensor([0]).float().cuda(), y_sim), dim=0)

    return torch.logsumexp(y_sim, dim=0)


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
            """
            train_text.append(t1)
            train_text2.append(t1)
            label.append(1)
            train_text.append(t2)
            train_text2.append(t2)
            label.append(1)
            train_text.append(t1)
            train_text2.append(train_text2[random.randint(0, len(train_text2)-4)])
            label.append(0)
            train_text.append(t2)
            train_text2.append(train_text[random.randint(0, len(train_text)-5)])
            label.append(0)
            """
    reader = csv.reader(open("test.tsv", "r", encoding="utf-8"), delimiter='\t')
    for id, t1, t2 in reader:
        if not id.isalpha():
            test_text.append(t1)
            test_text2.append(t2)

    test_set = TextDataset(test=True, text=test_text, text2=test_text2)
    train_set = TextDataset(test=False, text=train_text, text2=train_text2, label=label)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=8, drop_last=True)
    device = torch.device('cuda')
    model = CoSent().to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    total_steps = len(train_loader) * Epoch
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    # criterion = nn.CrossEntropyLoss().to(device)
    print("Start training!")

    # train
    batch_count = len(train_text) // batch_size
    model.train()
    for epoch in range(Epoch):
        print_avg_loss = 0
        for batch_idx, ((x, mx, tx), (x2, mx2, tx2), label) in enumerate(train_loader):
            x = x.to(device)
            mx = mx.to(device)
            tx = tx.to(device)
            x2 = x2.to(device)
            mx2 = mx2.to(device)
            tx2 = tx2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out1 = model(x, mx, tx)
            out2 = model(x2, mx2, tx2)
            sim = compute_sim(out1, out2)
            loss = compute_loss(label, sim)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            print_avg_loss += loss.item()
        scheduler.step()
        print("Epoch: %d, Loss: %.4f" % ((epoch + 1), print_avg_loss / batch_count))

    # find best threshold
    model.eval()
    preds = np.array([])
    trues = np.array([])
    with torch.no_grad():
        for batch_idx, ((x, mx, tx), (x2, mx2, tx2), label) in enumerate(train_loader):
            x = x.to(device)
            mx = mx.to(device)
            tx = tx.to(device)
            x2 = x2.to(device)
            mx2 = mx2.to(device)
            tx2 = tx2.to(device)
            out1 = model(x, mx, tx)
            out2 = model(x2, mx2, tx2)
            sim = compute_sim(out1, out2)
            preds = np.append(preds, sim.cpu().numpy())
            trues = np.append(trues, label.cpu().numpy())

    max_acc = 0
    threshold = 0
    for i in range(20, 80):
        th = i / 100.0
        acc = 0
        for j in range(len(preds)):
            if preds[j] >= th:
                acc += int(1 == trues[j])
            else:
                acc += int(0 == trues[j])
        if acc > max_acc:
            max_acc = acc
            threshold = th
    print(threshold)

    # test
    model.eval()
    preds = np.array([])
    with torch.no_grad():
        for batch_idx, ((x, mx, tx), (x2, mx2, tx2), _) in enumerate(test_loader):
            x = x.to(device)
            mx = mx.to(device)
            tx = tx.to(device)
            x2 = x2.to(device)
            mx2 = mx2.to(device)
            tx2 = tx2.to(device)
            out1 = model(x, mx, tx)
            out2 = model(x2, mx2, tx2)
            sim = compute_sim(out1, out2)
            preds = np.append(preds, sim.cpu().numpy())
    preds = preds.astype(float)

    # output
    writer = csv.writer(open("CoSent_submission.csv", "w", encoding="utf-8"))
    writer.writerow(("Id", "Category"))
    for i in range(len(preds)):
        if preds[i] >= threshold:
            writer.writerow((i, 1))
        else:
            writer.writerow((i, 0))


if __name__ == '__main__':
    main()
