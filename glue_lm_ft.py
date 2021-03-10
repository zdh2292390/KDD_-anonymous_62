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

from transformers import (
  BertConfig,
  BertForMaskedLM,
  BertTokenizer,
  GPT2Config,
  GPT2LMHeadModel,
  GPT2Tokenizer,
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

MODEL_CLASSES = {
  'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
  "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


def load_and_cache_examples(args, task, tokenizer, evaluate=False, return_features=False):
    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = 'classification'
    # Load data features from cache or dataset file
    # genre = args.tar_genre if evaluate else args.src_genre
    cached_features_file = "cached_{}_{}_{}_{}".format(
        "dev" if evaluate else "train",
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length),
        str(task),
    )
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    if os.path.exists(cached_features_file) and (not hasattr(args, 'overwrite_cache') or not args.overwrite_cache):
        print("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s" % args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else
            processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if not hasattr(args, 'local_rank') or args.local_rank in [-1, 0]:
            print("Saving features into cached file %s" % cached_features_file)
            torch.save(features, cached_features_file)

    if (hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]) and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if return_features:
        return features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def load_bert_dataset(args, task, tokenizer, evaluate=False, replica=1):
  features = load_and_cache_examples(args, task, tokenizer, evaluate, return_features=True)
  # in: input_ids, attention_mask, token_type_ids, label
  # out: input_ids, attention_mask, token_type_ids, masked_lm_labels
  all_input_ids = []
  all_attention_masks = []
  all_token_type_ids = []
  all_masked_lm_labels = []
  masked_lm_prob = 0.15
  max_predictions_per_seq = 20
  rng = random.Random(args.seed)
  MASK_id = tokenizer.mask_token_id
  for (ft_index, feature) in enumerate(features):
    for _ in range(replica):
      init_ids = feature.input_ids
      input_len = sum(feature.attention_mask)
      sep1_index = input_len - sum(feature.token_type_ids) - 1

      masked_lm_labels = [-100] * args.max_seq_length

      # sep1
      cand_indexes = list(range(1, sep1_index)) + list(range(sep1_index + 1, input_len - 1))
      rng.shuffle(cand_indexes)
      len_cand = len(cand_indexes)

      input_ids = copy.copy(init_ids)
      num_to_predict = min(max_predictions_per_seq,
                           max(1, int(round(input_len * masked_lm_prob))))

      masked_lms_pos = []
      covered_indexes = set()
      for index in cand_indexes:
        if len(masked_lms_pos) >= num_to_predict:
          break
        if index in covered_indexes:
          continue
        covered_indexes.add(index)

        masked_token = None
        if rng.random() < 0.8:
          masked_token = MASK_id
        else:
          if rng.random() < 0.5:
            masked_token = init_ids[index]
          else:
            masked_token = init_ids[cand_indexes[rng.randint(0, len_cand - 1)]]

        masked_lm_labels[index] = init_ids[index]
        input_ids[index] = masked_token
        masked_lms_pos.append(index)

      assert len(input_ids) == args.max_seq_length
      assert len(masked_lm_labels) == args.max_seq_length
      assert len(feature.attention_mask) == args.max_seq_length
      assert len(feature.token_type_ids) == args.max_seq_length

      if ft_index < 0:
        print("*** Example ***")
        print(" tokens: %s" % " ".join([str(x) for x in tokenizer.convert_ids_to_tokens(init_ids)]))
        print(" init_ids: %s" % " ".join([str(x) for x in init_ids]))
        print(' masked tokens: %s' % ' '.join([str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
        print(" input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print(" attention_mask: %s" % " ".join([str(x) for x in feature.attention_mask]))
        print(" token_type_ids: %s" % " ".join([str(x) for x in feature.token_type_ids]))
        print(" masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
        
      all_input_ids.append(input_ids)
      all_attention_masks.append(feature.attention_mask)
      all_token_type_ids.append(feature.token_type_ids)
      all_masked_lm_labels.append(masked_lm_labels)

  all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
  all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
  all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
  all_masked_lm_labels = torch.tensor(all_masked_lm_labels, dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_masked_lm_labels)

  return dataset


def load_and_cache_dataset(args, task, tokenizer, evaluate=False, replica=1):
  if args.model_type == 'bert':
    return load_bert_dataset(args, task, tokenizer, evaluate=evaluate, replica=replica)
  elif args.model_type == 'gpt2':
    return load_gpt2_dataset(args, task, tokenizer, evaluate=evaluate)
  else:
    raise NotImplementedError


def get_loss(model_type, model, criterion, batch, evaluate=False):
  if model_type == 'bert':
    if not evaluate:
      loss = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], masked_lm_labels=batch[3])[0]
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

      if global_step in ckpt_steps:
        ckpt_path = os.path.join(args.output_dir, f'step_{global_step}.bin')
        torch.save(model, ckpt_path)

        if args.do_eval:
          step_eval_results = evaluate(args, model, criterion, tokenizer)
          tr_loss = np.mean(step_loss)
          if len(step_eval_results) == 2:
            eval_loss = step_eval_results['mnli']
            eval_loss_mm = step_eval_results['mnli-mm']
            eval_results.append([global_step, tr_loss, eval_loss, eval_loss_mm])
            print(f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {tr_loss:.2f}, eval_loss = {eval_loss:.2f}, '
                  f'eval_loss_mm = {eval_loss_mm:.2f}\n')
          else:
            eval_loss = step_eval_results['mnli']
            eval_results.append([global_step, tr_loss, eval_loss])
            print(f'\nSaving model checkpoint to {ckpt_path}, avg_loss = {tr_loss:.2f}, eval_loss = {eval_loss:.2f}\n')
        else:
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


def evaluate(args, model, criterion, tokenizer):
  if not args.tar_genre:
    eval_task_names = ('mnli', 'mnli-mm') if args.task_name == 'mnli' else (args.task_name,)
  else:
    eval_task_names = (args.task_name,)

  model.eval()
  results = {}
  for eval_task in eval_task_names:
    eval_dataset = load_and_cache_dataset(args, eval_task, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_desc = eval_task
    if eval_task.startswith('mnli'):
      if args.tar_genre is not None and args.tar_genre != '':
        eval_desc = eval_task + '-' + args.tar_genre

    if args.model_type == 'bert':
      eval_loss = 0
      num_elements = 0
      with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f'eval: {eval_desc}', leave=False):
          # in: input_ids, attention_mask, token_type_ids, masked_lm_labels
          batch = tuple(t.to(args.device) for t in batch)
          loss = get_loss(args.model_type, model, criterion, batch, evaluate=True)
          eval_loss += loss.item()
          num_elements += (batch[3].detach().cpu().numpy().flatten() != criterion.ignore_index).sum()
        eval_loss /= num_elements
      results[eval_task] = eval_loss

  return results


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='data/Electronics', type=str)
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
  parser.add_argument('--baseon_bert', action='store_true')
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
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

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
    with open(args.data_dir+'/top1000.json', 'r') as inputfile:
      new_tokens=json.load(inputfile)
    new_tokens = new_tokens.keys()
    print('num of new tokens:'+str(len(new_tokens)))
    tokenizer.add_tokens(new_tokens=new_tokens)
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    
    if not args.baseon_bert:
      model = model_class(config=config)
    else:
      print("pretrain based on bert")
      model = model_class.from_pretrained(args.model_name_or_path)

    # if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    # eval_res = evaluate(args, model, criterion, tokenizer)
    # print(eval_res)
    # exit(0)

    # model.config.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    args_dict = copy.copy(args.__dict__)
    del args_dict['device']
    json.dump(args_dict, open(os.path.join(args.output_dir, 'args.json'), 'w'), ensure_ascii=False, indent=2)

    train_dataset = load_and_cache_dataset(args, args.task_name, tokenizer, evaluate=False)
    train(args, train_dataset, model, criterion, tokenizer)

  if not args.do_train and args.do_eval:
    config = config_class.from_pretrained(args.output_dir)
    tokenizer = get_tokenizer(args.model_type, args.output_dir)
    criterion = get_criterion(args.model_type, tokenizer)
    criterion.to(args.device)

    ckpt_paths = [f for f in os.listdir(args.output_dir) if f.startswith('step_') and f.endswith('.bin')]
    ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.strip('step_').strip('.bin')))
    eval_results = []

    for ckpt_path in ckpt_paths:
      model = torch.load(os.path.join(args.output_dir, ckpt_path))
      model.to(args.device)
      ckpt_step = int(ckpt_path.strip('step_').strip('.bin'))
      step_eval_results = evaluate(args, model, criterion, tokenizer)

      if len(step_eval_results) == 2:
        eval_loss = step_eval_results['mnli']
        eval_loss_mm = step_eval_results['mnli-mm']
        eval_results.append([ckpt_step, eval_loss, eval_loss_mm])
        print(f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}, eval_loss_mm = {eval_loss_mm:.2f}\n')
      else:
        eval_loss = step_eval_results['mnli']
        eval_results.append([ckpt_step, eval_loss])
        print(f'\nCheckpoint = {ckpt_path}, eval_loss = {eval_loss:.2f}\n')

    if len(eval_results[0]) == 3:
      header = ['step', 'eval_loss', 'mm_loss']
    else:
      header = ['step', 'eval_loss']
    best_results = report_results(header, eval_results, 1)
    best_step = best_results[0]
    print(f'best_ckpt = {os.path.join(args.output_dir, f"step_{best_step}.bin")}\n')
