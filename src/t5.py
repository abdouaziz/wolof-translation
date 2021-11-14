import torch
import torch.nn as nn 
import argparse
import numpy as np
from torch.utils.data import DataLoader , Dataset
import pandas as pd 
from tqdm import tqdm 
from transformers import ( 
    BertTokenizer,
    AdamW ,  
    get_linear_schedule_with_warmup  ,
    T5Tokenizer,
    T5ForConditionalGeneration)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    parser = argparse.ArgumentParser(description="Pretrained Machine Translation French to Wolof")
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv  file containing the training data."
    )
 

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=150,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=150,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
         
    )

    parser.add_argument(
        "--number_epochs",
        type=int,
        default=3,
        help="Total number of training steps to perform the model .",
    ) 

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--epsilone",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    ) 

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-base",
        help="Pretrained model name.",
    )

    parser.add_argument(
        "--task_prefix",
        type=str,
        default="translate French to Wolof: ",
        help="The task prefix for the translation.",
    )
 
 
    args = parser.parse_args()

 
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
   
 
    return args
 
 


class NMTDataset(Dataset):
    def __init__(self, frenchs, wolofs , tokenizer , max_len_source , max_len_target):
        self.frenchs = frenchs
        self.wolofs = wolofs
        self.tokenizer = tokenizer
        self.max_len_source = max_len_source
        self.max_len_target = max_len_target
    
    
    def __len__(self):
        return len(self.frenchs)
    
    def __getitem__(self, item):
        
        french = str(self.frenchs[item])
        wolof = str(self.wolofs[item])

        french_encoding = self.tokenizer(
              french,
              add_special_tokens=True,
              max_length=self.max_len_source,
              pad_to_max_length=True,
              return_attention_mask=True,
              return_tensors='pt')    

        labels = self.tokenizer(
              wolof,
              add_special_tokens=True,
              max_length=self.max_len_target,
              return_token_type_ids=True,
              pad_to_max_length=True,
              return_attention_mask=True
              ).input_ids
        
        
        
        labels = [
           [(label if label != self.tokenizer.pad_token_id else -100) for label in labels]]

        labels = torch.tensor(labels)

        return {
                  'input_ids': french_encoding['input_ids'].flatten(),
                  'attention_mask':french_encoding['attention_mask'].flatten(),
                  'labels': labels.flatten()
                }





def NMTDataloader(df , batch_size , tokenizer , max_len_source , max_len_target):

    dataset = NMTDataset(df.french.values , df.wolof.values , tokenizer , max_len_source , max_len_target)
    dataloader = DataLoader(dataset , batch_size , num_workers= 4)
    return dataloader

 

def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_parameters, lr=3e-5, eps=1e-8)


def train_epoch (model , data_loader, optimizer , device , scheduler):
    model.train()
    losses = []
   
    for step , d in tqdm(enumerate(data_loader) , total=len(data_loader)):
        
        input_ids =d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)
       
       
        outputs = model(input_ids=input_ids , attention_mask=attention_mask , labels=labels)
        loss =  outputs.loss

        losses.append(loss.item())


        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (step + 1) % 10 == 0:
            
             
            print('Epoch: {} | loss: {} '.format(step+1, np.mean(losses)))


def train():

    args = parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    model= T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    df = pd.read_csv(args.train_file)
    df["french"]=df["french"].apply(lambda x:args.task_prefix +x)



    train_data_loader= NMTDataloader(df,args.train_batch_size , tokenizer , args.max_source_length , args.max_target_length) 
    
    nb_train_steps = int(len(train_data_loader) /args.train_batch_size * args.number_epochs)
    optimizer = yield_optimizer(model)
    scheduler = get_linear_schedule_with_warmup(
                                        optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=nb_train_steps)
    
    for epoch in range(args.number_epochs):
        
        print(f'Epoch {epoch + 1}')

        train_epoch(model,train_data_loader,optimizer,device,scheduler) 
        
     
    return model.save_pretrained("../model/bert2bert")


if __name__ == '__main__':
    train()