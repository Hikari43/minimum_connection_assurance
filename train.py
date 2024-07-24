import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    count = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            count += torch.tensor([data.size()[0]])
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(
        model, loss, optimizer, scheduler, 
        train_loader, val_loader, test_loader, 
        device, epochs, verbose):
    if val_loader != None:
        val_loss, val_acc1, val_acc5 = eval(
            model=model, loss=loss, dataloader=val_loader, 
            device=device, verbose=verbose)
    else:
        val_loss, val_acc1, val_acc5 = np.nan, np.nan, np.nan
    test_loss, test_acc1, test_acc5 = eval(
        model=model, loss=loss, dataloader=test_loader, 
        device=device, verbose=verbose)
    
    rows = [[np.nan, val_loss, val_acc1, val_acc5, test_loss, test_acc1, test_acc5]]
    
    for epoch in tqdm(range(epochs)):
        train_loss = train(
            model=model, loss=loss, optimizer=optimizer, dataloader=train_loader, 
            device=device, epoch=epoch, verbose=verbose)
        if val_loader != None:
            val_loss, val_acc1, val_acc5 = eval(
                model=model, loss=loss, dataloader=val_loader, device=device, verbose=verbose)
        else:
            val_loss, val_acc1, val_acc5 = np.nan, np.nan, np.nan
        test_loss, test_acc1, test_acc5 = eval(
            model=model, loss=loss, dataloader=test_loader, device=device, verbose=verbose)

        row = [train_loss, val_loss, val_acc1, val_acc5, test_loss, test_acc1, test_acc5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'val_loss', 'val_top1_acc', 'val_top5_acc', 'test_loss', 'test_top1_acc', 'test_top5_acc']
    return pd.DataFrame(rows, columns=columns)


