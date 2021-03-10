import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import copy
import torch.nn as nn
from PhraseExtractor import *
# import multiprocessing as mp
import pickle
from transformers import (
  BertConfig,
  BertForMaskedLM,
  BertForCrossEntityAlignment,
  BertTokenizer,
)
from data_utils import glue_processors as processors
from data_utils import glue_convert_examples_to_features as convert_examples_to_features
from data_utils import InputFeatures
from print_hook import redirect_stdout
from utils import get_adamw, report_results, get_tokenizer, get_criterion, set_seed

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

# Pretraining head for CrossEntityAlignment is defined in "./transformers/modeling_bert.py"
MODEL_CLASSES = {'bert': (BertConfig, BertForCrossEntityAlignment, BertTokenizer),}


def load_and_cache_examples(args, task, tokenizer, evaluate=False, return_features=False):
    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = 'classification'
    # Load data features from cache or dataset file
    cached_features_file = "cached_{}_{}_{}_{}".format(
        "dev" if evaluate else "train",
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length),
        str(task),
    )
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    cached_phrase_file = os.path.join(args.data_dir, 'phrase_file')
    if os.path.exists(cached_features_file) and (not hasattr(args, 'overwrite_cache') or not args.overwrite_cache):
        print("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
        with open(cached_phrase_file, 'rb') as fp:
          Phrases = pickle.load(fp)
    else:
        print("Creating features from dataset file at %s" % args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm", "ahm", "cea"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else
            processor.get_train_examples(args.data_dir)
        )
        features, Phrases = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
			task = task,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
          phrase_table= get_phrase_table()['phrase'].unique()
        )
        if not hasattr(args, 'local_rank') or args.local_rank in [-1, 0]:
            print("Saving features into cached file %s" % cached_features_file)
            with open(cached_phrase_file, 'wb') as fp:
              pickle.dump(Phrases, fp)
            torch.save(features, cached_features_file)

    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if return_features:
        return features, Phrases

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	if task = 'cea':
		all_len_entity1 = torch.tensor([f.len_entity1 for f in features], dtype=torch.long)
		all_len_entity2 = torch.tensor([f.len_entity2 for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

	if task = 'cea':
    	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_len_entity1, all_len_entity2)
	else:
		dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

  	return dataset


def load_bert_dataset(args, task, tokenizer, evaluate=False, replica=1):
  features, Phrases = load_and_cache_examples(args, task, tokenizer, evaluate, return_features=True)
  return features


def load_and_cache_dataset(args, task, tokenizer, evaluate=False, replica=1):
  if args.model_type == 'bert':
    return load_bert_dataset(args, task, tokenizer, evaluate=evaluate, replica=replica)
  else:
    raise NotImplementedError

def get_loss(model_type, model, criterion, batch, evaluate=False):
  if model_type == 'bert':
    if not evaluate:
      loss = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], masked_lm_labels=batch[3], len_entity1=batch[-2], len_entity2=batch[-1],)[0]
	else:
      prediction_scores = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])[0]
      loss = criterion(prediction_scores.view(-1, model.config.vocab_size), batch[3].view(-1))
    return loss
  else:
    raise NotImplementedError

def train(args, train_dataset, model, criterion, tokenizer):
  tb_writer = SummaryWriter(args.output_dir)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
  else:
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    args.num_train_epochs = int(np.ceil(args.num_train_epochs))

  optimizer, scheduler = get_adamw(model, t_total, args.warmup_steps, args.learning_rate, weight_decay=args.weight_decay)

  train_desc = args.task_name
  if args.src_genre is not None and args.src_genre != '':
    train_desc += '-' + args.src_genre
  print(f'***** Fine-tuning {args.model_name_or_path} {train_desc} *****')
  print(f'  Num examples = {len(train_dataset)}')
  print(f'  Num Epochs = {args.num_train_epochs}')
  print(f'  Train batch size = {args.train_batch_size}')
  print(f'  Total optimization steps = {t_total}')

  ckpt_steps = set([int(x) for x in np.linspace(0, t_total, args.num_ckpts + 1)[1:]])

  model.train()
  model.zero_grad()

  global_step = 0
  step_loss = []
  eval_results = []

  pbar = tqdm(total=t_total, desc=f'train')
  set_seed(args)
  for epoch in range(args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
      model.train()
      batch = tuple(t.to(args.device) for t in batch)
      loss = get_loss(args.model_type, model, criterion, batch)
      loss.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()
      step_loss.append(loss.item())
      global_step += 1
      pbar.update(1)
      pbar.set_description_str(f'train: {train_desc} (loss = {step_loss[-1]:.2f}, lr = {scheduler.get_lr()[0]:.2g})')

      print(global_step, ckpt_steps, args.do_eval)
      if global_step not in ckpt_steps:
        ckpt_path = os.path.join(args.output_dir, f'step_{global_step}.bin')
        torch.save(model, ckpt_path)        
        print(f'\nSaving model checkpoint to {ckpt_path}\n')

      if global_step % args.logging_steps == 0:
        tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', np.mean(step_loss), global_step)
        step_loss = []

      if global_step == args.max_steps:
        pbar.close()
        break

  if args.do_eval:
    if len(eval_results[0]) == 4:
      header = ['step', 'avg_loss', 'eval_loss', 'mm_loss']
    else:
      header = ['step', 'avg_loss', 'eval_loss']
    best_results = report_results(header, eval_results, 2)
    best_step = best_results[0]
    print(f'best_ckpt = {os.path.join(args.output_dir, f"step_{best_step}.bin")}\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='data/Electronics', type=str)
  parser.add_argument('--data_id', default=1, type=int)
  parser.add_argument('--model_type', default='bert', type=str, help='gpt2')
  parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str, help='gpt2')
  parser.add_argument('--task_name', default='mnli')
  parser.add_argument('--src_genre', default='', type=str)
  parser.add_argument('--tar_genre', default='', type=str)
  parser.add_argument('--output_dir', default='', type=str, required=True)
  parser.add_argument('--config_name', default='', type=str)
  parser.add_argument('--tokenizer_name', default='', type=str)
  parser.add_argument('--max_seq_length', default=128, type=int)
  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_eval', action='store_true')
  parser.add_argument('--train_batch_size', default=16, type=int)
  parser.add_argument('--eval_batch_size', default=64, type=int)
  parser.add_argument('--learning_rate', default=5e-5, type=float)
  parser.add_argument('--weight_decay', default=0.0, type=float)
  parser.add_argument('--adam_epsilon', default=1e-8, type=float)
  parser.add_argument('--max_grad_norm', default=1.0, type=float)
  parser.add_argument('--num_train_epochs', default=1.0, type=float)
  parser.add_argument('--max_steps', default=-1, type=int)
  parser.add_argument('--warmup_steps', default=0, type=int)
  parser.add_argument('--logging_steps', type=int, default=100)
  parser.add_argument('--num_ckpts', type=int, default=10)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--baseon_bert', default=True, action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')
  args.device = device

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  log_file = open(os.path.join(args.output_dir, 'log'), 'a')
  redirect_stdout(log_file)
  set_seed(args)

  args.task_name = args.task_name.lower()
  if args.task_name not in processors:
    raise ValueError('Task not found: %s' % (args.task_name))
  processor = processors[args.task_name]()
  args.model_type = args.model_type.lower()

  config_class, model_class, tokenizer_class = BertConfig, BertForCrossEntityAlignment, BertTokenizer
  print('Training/evaluation parameters %s' % str(args))

  if args.do_train:
    config = config_class.from_pretrained(
      args.config_name or args.model_name_or_path,
      finetuning_taks=args.task_name,
    )

    tokenizer = get_tokenizer(args.model_type, args.tokenizer_name or args.model_name_or_path)
    criterion = get_criterion(args.model_type, tokenizer)
    print(f'*** Criterion ignore_index = {criterion.ignore_index} ***')
    criterion.to(args.device)

    model = model_class(config=config)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.save_pretrained(args.output_dir)

    args_dict = copy.copy(args.__dict__)
    del args_dict['device']
    json.dump(args_dict, open(os.path.join(args.output_dir, 'args.json'), 'w'), ensure_ascii=False, indent=2)

    train_dataset = load_and_cache_dataset(args, args.task_name, tokenizer, evaluate=False)
    train(args, train_dataset, model, criterion, tokenizer)

