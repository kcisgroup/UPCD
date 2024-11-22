import torch
import random
from collections import Counter
from transformers import BertForMaskedLM
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
from my_models.symbolization import Symbolization


class ClusterLayer(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.dim = config.hidden_size * args.feature_dim
        self.codebook_dim = config.hidden_size * args.feature_dim
        self.codebook_size = args.cluster_space
        self.cluter_layer = VectorQuantize(dim=self.dim, codebook_dim=self.codebook_dim, codebook_size=self.codebook_size)

    def forward(self, hidden_states):
        batch, dim = hidden_states.shape
        quantize, embed_ind, loss = self.cluter_layer(hidden_states)

        return quantize, embed_ind, loss

class BCM(nn.Module):
    def __init__(self, args):
        super(BCM, self).__init__()

        self.device = torch.device(f'cuda:{args.gpu_id}' if args.cuda else 'cpu')
        self.model_path = args.lm_path

        self.args = args
        self.symbol_layer = Symbolization(self.args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.config = BertConfig.from_pretrained(self.model_path)
        self.config.vocab_size = len(self.tokenizer)

        self.bert_model = AutoModelForMaskedLM.from_config(self.config)
        self.cluster_layer = ClusterLayer(self.args, self.config)




    def forward(self, x):
        output, symbol_loss, encoding_one_hot = self.symbol_layer(x)
        encoding_one_hot = encoding_one_hot.to(torch.int)
        # sen_lists:获取一个个字符组成的句子
        sen_lists, custom_vocab = self.convert_encoding_to_sens(encoding_one_hot)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # 建立一个字典 字符对应的下标
        vocab_dict = self.convert_custom_vocab_to_vocab_dict(custom_vocab, tokenizer)
        # 将字符句子转换为下标
        sen_ids = self.convert_sens_to_ids(sen_lists, tokenizer)
        inputs = self.mask_ids(sen_ids, vocab_dict, tokenizer, max_length=self.args.feature_dim)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 调整此嵌入层大小以适应新词汇表
        self.bert_model.resize_token_embeddings(len(tokenizer))
        vocab_size = len(tokenizer)
        self.config.vocab_size = vocab_size
        tokenizer.save_pretrained(self.model_path)

        outputs = self.bert_model(**inputs, output_hidden_states=True)
        # 获取损失
        mlm_loss = outputs.loss
        hidden_states = outputs.hidden_states
        last_hidden_states = hidden_states[-1]
        batch_size, seq, dim = last_hidden_states.shape

        cls_vectors = last_hidden_states.view((batch_size, -1))

        cls_encoding, cluster_ids, cluster_loss = self.cluster_layer(cls_vectors)

        loss =  symbol_loss + mlm_loss + cluster_loss * self.args.alpha

        return  loss, cluster_ids

    def convert_encoding_to_sens(self, encoding_one_hots):
        sen_lists = []
        encoding_one_hots = encoding_one_hots.tolist()
        custom_vocab = []
        for i in range(len(encoding_one_hots)):
            sen = []
            for j in range(len(encoding_one_hots[0])):
                token = ""
                token += str(j)
                for num in encoding_one_hots[i][j]:
                    token += str(num)
                if token not in custom_vocab:
                    custom_vocab.append(token)
                sen.append(token)
            sen_lists.append(sen)
        return sen_lists, custom_vocab

    def convert_custom_vocab_to_vocab_dict(self, custom_vocab, tokenizer):
        vocab_dict = {}
        for i in custom_vocab:
            vocab_dict[i] = tokenizer.convert_tokens_to_ids(i)
        vocab_dict['[MASK]'] = tokenizer.convert_tokens_to_ids('[MASK]')
        return vocab_dict

    def convert_sens_to_ids(self, sens, tokenizer):
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
        input_ids = []
        for i in range(len(sens)):
            # one_ids = [cls_id]
            one_ids = []
            for j in sens[i]:
                one_ids.append(tokenizer.convert_tokens_to_ids(j))
            # one_ids.append(sep_id)
            input_ids.append(one_ids)
        return input_ids

    def mask_ids(self, sen_ids, vocab_dict, tokenizer, max_length=60, mask_prob=0.15):
        input_ids = sen_ids
        for i in range(len(input_ids)):
            input_ids[i] = self.pad_and_truncate(input_ids[i], max_length)

        input_ids = torch.tensor(input_ids)
        batch, seq_len = input_ids.shape
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # labels = torch.full(input_ids.shape, -100)
        labels = input_ids.clone()
        masked_indices = torch.zeros_like(input_ids)

        for i in range(batch):
            sen_length = input_ids[i].numel()
            max_indices_to_mask = max(1, int(sen_length * 0.15))

            # 生成随机下标，确保不重复且在有效范围内
            if max_indices_to_mask == 1:
                # 如果只需要选1个，直接随机选一个
                rand_indices = random.randint(1, sen_length - 1)
            else:
                # 如果需要选多个，确保不重复
                rand_indices = random.sample(range(1, sen_length), max_indices_to_mask)

            masked_indices[i, rand_indices] = 1
            labels[i, rand_indices] = input_ids[i, rand_indices]
            input_ids[i, rand_indices] = vocab_dict['[MASK]']

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        return inputs


    def pad_and_truncate(self, sequence, max_length):
        sequence = list(sequence)
        while len(sequence) < max_length:
            sequence.append(0)  # 添加填充 token
        if len(sequence) > max_length:
            sequence = sequence[:max_length]  # 截断过长的部分
        return sequence


    def loss_function(self, output, input):
        recons_loss = F.mse_loss(output, input)
        return recons_loss

