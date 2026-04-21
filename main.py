import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from lotr.optim import LoTRAdamW

from utils.make_lotr import make_lotr
from utils.make_lotr4 import make_lotr4
from utils.evaluate import evaluate_model
from utils.parameters import print_trainable_params
from utils.load_data import load_and_preprocess_data, load_roberta
from utils.device import device, autocast, grad_scaler

from typing import List, Tuple, Dict, Union, Optional, Literal
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(precision=8, linewidth=300)


def train(task_name, maker: Literal['lotr', 'lotr4'] = 'lotr'):
    RANK        = 64
    RANKS       = (64, 4, 16)
    LR          = 1e-3
    NUM_EPOCHS  = 20
    BATCH_SIZE  = 128
    
    # 1. 加载数据
    train_dataloader, eval_dataloader, num_labels = load_and_preprocess_data(BATCH_SIZE, task_name)
    
    # 2. 加载模型
    model = load_roberta(num_labels)
    
    # 3. 注入 LoTR
    if maker == 'lotr':
        model = make_lotr(model, rank=RANK, use_svd_init=True)
    elif maker == 'lotr4':
        model = make_lotr4(model, rank=RANKS, core_init='trivial', factor_init='svd')
    else:
        raise ValueError(f'不支持的注入: {maker}')
    
    # 4. 解冻分类器
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model.to(device)
    print_trainable_params(model)

    # 5. 配置优化器
    lotr_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        # print(f'{f"{name}":100s} | {f"{param.requires_grad}":10s} | {f"{param.shape}":40s} | {f"{param.dtype}":20s}')
        if not param.requires_grad:
            continue
        # 判断是否属于 LoTR 或 LoTR3 层 (名字里包含 lotr 或 lotr3)
        if 'lotr' in name:
            lotr_params.append(param)
        else:
            head_params.append(param)
    
    # exit()
    # for param in lotr_params:
    #     print(f'{f"{param.shape}":40s} | {f"{param.dtype}":20s}')
    
    opt_lotr = AdamW(lotr_params, lr=LR)
    opt_head = AdamW(head_params, lr=LR)
    
    # 6. 训练循环
    acc_before = evaluate_model(model, eval_dataloader, device)
    print(f"Pre-train Accuracy: {acc_before:.16f}")

    writer = SummaryWriter(f'./running_logs/{task_name}/{maker}')
    writer.add_scalar('acc', acc_before, 0)
    
    history = []
    for epoch in range(1, 1 + NUM_EPOCHS):
        model.train()

        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Task {task_name} | Epoch {epoch}")
        
        for batch in progress_bar:
            batch = {k:v.to(device) for k,v in batch.items()}
            
            opt_lotr.zero_grad()
            opt_head.zero_grad()

            with autocast:
                outputs = model(**batch)
                loss: torch.Tensor = outputs.loss

            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt_lotr)
            grad_scaler.step(opt_head)
            grad_scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        acc = evaluate_model(model, eval_dataloader, device)

        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('acc', acc, epoch)

        history.append(acc)
        print(f"Epoch {epoch} Val Acc: {acc:.16f}")
    
    writer.close()
    return history[-1]

def main():
    tasks = [
        'cola',
        'mnli',
        'mrpc',
        'qnli',
        'qqp',
        'rte',
        'sst2',
    ]
    makers = [
        'lotr4',
        'lotr',
    ]
    
    for maker in makers:
        results = {}
        for task in tasks:
            print(f"\n{'='*10} Processing {task} {'='*10}")
            final_acc = train(task, maker=maker)
            results[task] = final_acc
        print("\nFinal Results:", results)

if __name__ == '__main__':
    main()
