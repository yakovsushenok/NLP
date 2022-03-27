import torch
import time

def model_train(config,net,criterion,optimizer,train_dataloader,epoch):

    train_res = None
    net.train()
    total_acc, total_count = 0, 0
    log_interval = 250

    for idx, mini_batch in enumerate(train_dataloader):

        #train model
        optimizer.zero_grad()
        predicted_labels = net(mini_batch['lyrics'],mini_batch['text_len'])
        task_targets = {task:mini_batch[task]for task in config["Tasks"]}
        loss = criterion(predicted_labels, task_targets)
        loss['total'].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        optimizer.step()

        #calc metrics
        total_acc += (predicted_labels['genre'].argmax(1) == mini_batch['genre']).sum().item()
        total_count += mini_batch['genre'].size(0)


        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(train_dataloader),
                                              total_acc/total_count))
            #train_res = update_train_dict_(config,total_acc/total_count,train_res,)
            total_acc, total_count = 0, 0


    return net


