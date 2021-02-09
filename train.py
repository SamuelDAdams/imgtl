import torch
import torchvision
import torch.nn as nn
from img2vec_pytorch import Img2Vec
import numpy as np
from scipy.linalg import sqrtm, inv
from simplenet import FC, Net2
import random
import os
from torch.optim import AdamW
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

def gen_transform(target_data):
    return sqrtm(np.cov(target_data, rowvar=False) + np.eye(target_data.shape[1]))

def CORAL(source, target):
    c_s = inv(gen_transform(source))
    c_t = gen_transform(target)
    d_s = np.matmul(source, c_s)
    d_s = np.matmul(d_s, c_t)
    return np.real(d_s)

def load(directory, label):
    embeds = []
    labels = []
    img2vec = Img2Vec(cuda=True)
    for file in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, file)):
            continue
        if not os.path.exists(directory + '/saves/' + file + '.npy'):
            if not os.path.exists(directory + '/saves'):
                os.makedirs(directory + '/saves')
            image = Image.open(directory + '/'+ file).convert('RGB')
            embed = img2vec.get_vec(image)
            np.save(directory + '/saves/' + file + '.npy', embed, allow_pickle=True)
            embeds.append(embed)
            labels.append(label)
            print('cmpl: {}'.format(directory + '/'+ file))
        else:
            embeds.append(np.load(directory + '/saves/' + file + '.npy'))
            labels.append(label)
    return embeds, labels

def load_and_process(directory, random_sampler=True, ttsplit_ratio=0, sample_amount=0, just_numpy=False):
    normal, norm_lab = load(directory + '/normal', 0)
    pneu, pneu_lab = load(directory + '/pneumonia', 1)
    lab = norm_lab + pneu_lab
    data = np.vstack((normal, pneu))
    if sample_amount > 0:
        selection = np.random.choice(data.shape[0], sample_amount, replace=False)
        data = data[selection]
        lab = np.asarray(lab)
        lab = lab[selection]
    if just_numpy:
        if ttsplit_ratio > 0:
            data_train, data_test, lab_train, lab_test = train_test_split(data, lab, test_size=ttsplit_ratio)
            return data_train, data_test, lab_train, lab_test
        else:
            return data, lab
    lab = torch.LongTensor(lab)
    data = torch.tensor(data)
    if ttsplit_ratio > 0:
        return make_loaders(data, lab, tts_ratio=ttsplit_ratio)
    return make_loaders(data, lab)

def make_loaders(data, labels, tts_ratio=0, random_sampler=True):
    if tts_ratio > 0:
        data_train, data_test, label_train, lab_test = train_test_split(data, labels, test_size=tts_ratio)
        train_data = TensorDataset(data_train, label_train)
        train_sampler = RandomSampler(train_data) 
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=50)
        test = TensorDataset(data_test, lab_test)
        test_sampler = SequentialSampler(test) 
        test_loader = DataLoader(test, sampler=test_sampler, batch_size=50)
        return train_loader, test_loader
    dataset = TensorDataset(data, labels)
    sampler = RandomSampler(dataset) if random_sampler else SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=50)
    return loader

def train(model, train_loader, val_loader, test_loader, epochs):
    print('Starting training')
    epoch_losses = []
    opt = AdamW(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        losses = []
        lossf = nn.BCEWithLogitsLoss()
        for _, batch in enumerate(train_loader):
            embed_gpu, lab_gpu = tuple(i.to('cuda') for i in batch)
            model.zero_grad()
            logits = model(embed_gpu)
            y = nn.functional.one_hot(lab_gpu, num_classes=2)
            loss = lossf(logits, y.float())
            losses.append(loss.item() / len(batch))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        epoch_loss = np.mean(losses)
        if epoch % 10 == 0:
            print('Epoch {}\nTraining Loss: {}'.format(epoch, epoch_loss))
            val_loss = validate_model(model, val_loader, 'Validation')
        epoch_losses.append((epoch, epoch_loss, val_loss))
    validate_model(model, test_loader, 'Test')
    return epoch_losses

def validate_model(model, loader, set_name):
    model.eval()
    acc = []
    losses = []
    for batch in loader:
        gpu_embed, gpu_lab = tuple(i.to('cuda') for i in batch)

        with torch.no_grad():
            logits = model(gpu_embed)
        lossf = nn.BCEWithLogitsLoss()
        y = nn.functional.one_hot(gpu_lab, num_classes=2)
        loss = lossf(logits, y.float())
        losses.append(loss.item()/len(batch))
        _, predictions = torch.max(logits, dim=1)
        accuracy = (predictions == gpu_lab).cpu().numpy().mean()
        acc.append(accuracy)
    print('{} loss: {}'.format(set_name, np.mean(losses)))
    print('{} acc: {}'.format(set_name, np.mean(acc)))
    return np.mean(losses)

#normal and pneumonia mapped to pneumonia and covid
print(torch.cuda.is_available())
adult_train, adult_val = load_and_process('adult/train', ttsplit_ratio=0.25)
adult_test = load_and_process('adult/test', random_sampler=False)

initial_model = Net2()
initial_model.to('cuda')
train(initial_model, adult_train, adult_val, adult_test, 100)

pediatric_test = load_and_process('pediatric/test', random_sampler=False)
validate_model(initial_model, pediatric_test, 'Pediatric no domain shift')
pediatric_train_X, pediatric_train_y = load_and_process('pediatric/train', just_numpy=True, sample_amount=200)
adult_train_X, adult_train_y = load_and_process('adult/train', just_numpy=True)
adult_train_X = CORAL(adult_train_X, pediatric_train_X)
pediatric_train_X = np.vstack((pediatric_train_X, adult_train_X))
pediatric_train_X = torch.FloatTensor(pediatric_train_X)
pediatric_train_y = np.concatenate((pediatric_train_y, adult_train_y))
pediatric_train_y = torch.LongTensor(pediatric_train_y)


ped_train, ped_val = make_loaders(pediatric_train_X, pediatric_train_y, tts_ratio=0.25)
ped_test = load_and_process('pediatric/test', random_sampler=False)

new_model = Net2()
new_model.to('cuda')
train(new_model, ped_train, ped_val, ped_test, 100)
