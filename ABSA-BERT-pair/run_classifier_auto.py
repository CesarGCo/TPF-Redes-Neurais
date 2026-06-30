# coding=utf-8

"""Runner genérico de fine-tuning para ABSA usando HuggingFace AutoModel.

Permite trocar o backbone (RoBERTa, DeBERTa-v3, etc.) apenas mudando o
parâmetro --model_name. Mantém a mesma lógica de par de sentenças do
BERT-pair original.
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimization import BERTAdam
from processor import (Semeval_NLI_B_Processor, Semeval_NLI_M_Processor,
                       Semeval_QA_B_Processor, Semeval_QA_M_Processor,
                       Semeval_single_Processor, Sentihood_NLI_B_Processor,
                       Sentihood_NLI_M_Processor, Sentihood_QA_B_Processor,
                       Sentihood_QA_M_Processor, Sentihood_single_Processor)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features_auto(examples, label_list, max_seq_length, tokenizer):
    """Tokeniza usando o tokenizer carregado por AutoTokenizer (genérico)."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if example.text_b:
            encoding = tokenizer(
                example.text_a,
                example.text_b,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
        else:
            encoding = tokenizer(
                example.text_a,
                max_length=max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )

        input_ids = encoding['input_ids']
        input_mask = encoding['attention_mask']
        # token_type_ids: alguns modelos fornecem, outros não. Usamos se houver.
        if 'token_type_ids' in encoding:
            segment_ids = encoding['token_type_ids']
        else:
            segment_ids = [0] * max_seq_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        choices=["sentihood_single", "sentihood_NLI_M", "sentihood_QA_M",
                                 "sentihood_NLI_B", "sentihood_QA_B", "semeval_single",
                                 "semeval_NLI_M", "semeval_QA_M", "semeval_NLI_B", "semeval_QA_B"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", type=str,
                        help="Nome ou caminho do modelo HuggingFace (ex: roberta-base, microsoft/deberta-v3-base).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--eval_test", default=False, action='store_true')
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=4, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--no_cuda", default=False, action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--accumulate_gradients', type=int, default=1)

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processors = {
        "sentihood_single": Sentihood_single_Processor,
        "sentihood_NLI_M": Sentihood_NLI_M_Processor,
        "sentihood_QA_M": Sentihood_QA_M_Processor,
        "sentihood_NLI_B": Sentihood_NLI_B_Processor,
        "sentihood_QA_B": Sentihood_QA_B_Processor,
        "semeval_single": Semeval_single_Processor,
        "semeval_NLI_M": Semeval_NLI_M_Processor,
        "semeval_QA_M": Semeval_QA_M_Processor,
        "semeval_NLI_B": Semeval_NLI_B_Processor,
        "semeval_QA_B": Semeval_QA_B_Processor,
    }

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    train_features = convert_examples_to_features_auto(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features_auto(
            test_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list)
    )
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=", output_log_file)
    with open(output_log_file, "w") as writer:
        if args.eval_test:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")

    global_step = 0
    epoch = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch += 1
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_ids
            )
            loss = outputs.loss

            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        if args.eval_test:
            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
            with open(os.path.join(args.output_dir, f"test_ep_{epoch}.txt"), "w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids,
                            labels=label_ids
                        )
                        tmp_test_loss = outputs.loss
                        logits = outputs.logits

                    logits = F.softmax(logits, dim=-1)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs_pred = np.argmax(logits, axis=1)
                    for output_i in range(len(outputs_pred)):
                        f_test.write(str(outputs_pred[output_i]))
                        for ou in logits[output_i]:
                            f_test.write(" " + str(ou))
                        f_test.write("\n")
                    tmp_test_accuracy = np.sum(outputs_pred == label_ids)

                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy
                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples

        result = collections.OrderedDict()
        if args.eval_test:
            result = {'epoch': epoch,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps,
                      'test_loss': test_loss,
                      'test_accuracy': test_accuracy}
        else:
            result = {'epoch': epoch,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")


if __name__ == "__main__":
    main()
