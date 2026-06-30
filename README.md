# ABSA-BERT-pair: Reprodução e Aprimoramento

Reprodução e aprimoramento do artigo **"Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence"** (Sun et al., NAACL 2019), aplicado ao dataset **SemEval-2014 Task 4**.

Trabalho desenvolvido para a disciplina de Redes Neurais (UFV, Ciência da Computação).

* **TPF-01:** reprodução do modelo BERT-pair-NLI-M original.
* **TPF-02:** aprimoramento via troca de *backbone* (BERT para RoBERTa e DeBERTa-v3) e estudo de ablação na cabeça de classificação.

Baseado no repositório original dos autores: [HSLCY/ABSA-BERT-pair](https://github.com/HSLCY/ABSA-BERT-pair).

---

## Resultados

Desempenho no SemEval-2014 Task 4 (variante NLI-M), todos com 4 épocas, `lr=2e-5`, `seed=42`:

| Modelo | Aspect F1 | Sentiment 4-cl. | Sentiment 3-cl. | Sentiment 2-cl. |
|---|---|---|---|---|
| Artigo original (BERT) | 91.67 | 85.10 | 88.70 | 95.10 |
| Reprodução TPF-01 (BERT) | 91.99 | 84.20 | 87.98 | 94.77 |
| RoBERTa-base | 91.70 | 87.61 | 91.26 | 96.70 |
| RoBERTa + camada oculta | 91.52 | 85.95 | 89.41 | 94.54 |
| **DeBERTa-v3-base** | **92.45** | **89.95** | **92.91** | **97.72** |

A troca do *backbone* trouxe ganho consistente, com o DeBERTa-v3 superando todos os modelos em todas as métricas. O aumento de capacidade da cabeça de classificação (camada oculta extra) não trouxe benefício, indicando que o gargalo do modelo original estava na qualidade do *backbone* pré-treinado.

---

## Ambiente

Testado em Windows 11 com GPU NVIDIA RTX 4060 Ti.

```bash
conda create -n absa python=3.7 -y
conda activate absa

pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-pretrained-bert==0.6.2
pip install transformers==4.18.0
pip install sentencepiece
pip install pandas scikit-learn
```

A versão `transformers==4.18.0` roda todos os modelos (RoBERTa e DeBERTa-v3). O `sentencepiece` é necessário para o tokenizador do DeBERTa-v3.

---

## Preparação dos dados

Os dados do SemEval-2014 já acompanham o repositório. Para gerar os arquivos no formato *sentence-pair*:

```bash
cd generate
python generate_semeval_NLI_M.py
python generate_semeval_QA_M.py
python generate_semeval_NLI_B_QA_B.py
python generate_semeval_BERT_single.py
cd ..
```

---

## Pesos pré-treinados

### BERT (TPF-01)

O link oficial do Google para os pesos do BERT está indisponível. Baixe os arquivos equivalentes do espelho no HuggingFace e converta o checkpoint para o formato esperado pelo código:

```bash
mkdir uncased_L-12_H-768_A-12
curl -L -o uncased_L-12_H-768_A-12/vocab.txt https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt
curl -L -o uncased_L-12_H-768_A-12/bert_config.json https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json
curl -L -o uncased_L-12_H-768_A-12/pytorch_model.bin https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin

python fix_checkpoint.py
```

O script `fix_checkpoint.py` remove o prefixo `bert.` e descarta as camadas de *masked language modeling*, adequando o checkpoint moderno ao formato esperado pela classe `BertModel` do repositório original.

### RoBERTa (TPF-02)

```bash
mkdir roberta-base
curl -L -o roberta-base/vocab.json https://huggingface.co/roberta-base/resolve/main/vocab.json
curl -L -o roberta-base/merges.txt https://huggingface.co/roberta-base/resolve/main/merges.txt
curl -L -o roberta-base/config.json https://huggingface.co/roberta-base/resolve/main/config.json
curl -L -o roberta-base/pytorch_model.bin https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
```

### DeBERTa-v3 (TPF-02)

```bash
mkdir deberta-v3-base
curl -L -o deberta-v3-base/config.json https://huggingface.co/microsoft/deberta-v3-base/resolve/main/config.json
curl -L -o deberta-v3-base/pytorch_model.bin https://huggingface.co/microsoft/deberta-v3-base/resolve/main/pytorch_model.bin
curl -L -o deberta-v3-base/spm.model https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model
curl -L -o deberta-v3-base/tokenizer_config.json https://huggingface.co/microsoft/deberta-v3-base/resolve/main/tokenizer_config.json
```

Os pesos são baixados manualmente porque a versão antiga da biblioteca `transformers` apresenta um problema ao montar a URL de download automático.

---

## Execução

### 1. Modelo original, BERT (TPF-01)

```bash
python run_classifier_TABSA.py --task_name semeval_NLI_M --data_dir data/semeval2014/bert-pair/ --vocab_file uncased_L-12_H-768_A-12/vocab.txt --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model_fixed.bin --eval_test --do_lower_case --max_seq_length 128 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 4 --output_dir results/semeval/NLI_M --seed 42
```

### 2. RoBERTa (TPF-02)

```bash
python run_classifier_TABSA_roberta.py --task_name semeval_NLI_M --data_dir data/semeval2014/bert-pair/ --roberta_model roberta-base --eval_test --max_seq_length 128 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 4 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir results/semeval/NLI_M_roberta --seed 42
```

### 3. DeBERTa-v3 (TPF-02, melhor modelo)

```bash
python run_classifier_auto.py --task_name semeval_NLI_M --data_dir data/semeval2014/bert-pair/ --model_name deberta-v3-base --eval_test --max_seq_length 128 --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 4 --warmup_proportion 0.1 --weight_decay 0.01 --output_dir results/semeval/NLI_M_deberta --seed 42
```

O *batch size* do DeBERTa-v3 foi reduzido para 16 devido ao maior consumo de memória do *disentangled attention*.

### 4. Ablação: RoBERTa + camada oculta (TPF-02)

```bash
python run_classifier_roberta_custom.py --task_name semeval_NLI_M --data_dir data/semeval2014/bert-pair/ --roberta_model roberta-base --eval_test --max_seq_length 128 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 4 --warmup_proportion 0.1 --weight_decay 0.01 --hidden_dim 256 --dropout_prob 0.3 --output_dir results/semeval/NLI_M_roberta_custom --seed 42
```

---

## Avaliação

Após o treino, calcule as métricas oficiais apontando para o arquivo de predições da época desejada:

```bash
python evaluation.py --task_name semeval_NLI_M --pred_data_dir results/semeval/NLI_M_deberta/test_ep_4.txt
```

---

## Modificações em relação ao código original

| Arquivo | Modificação |
|---|---|
| `tokenization.py` | Adicionado `encoding="utf-8"` na leitura do vocabulário (corrige erro de leitura no Windows). |
| `fix_checkpoint.py` | Script novo: adapta o checkpoint moderno do HuggingFace ao formato do código de 2019. |
| `run_classifier_TABSA_roberta.py` | Versão com *backbone* RoBERTa via biblioteca `transformers` e tokenização BPE. |
| `run_classifier_auto.py` | Runner genérico (classes `Auto*`), usado para o DeBERTa-v3 e extensível a outros modelos. |
| `run_classifier_roberta_custom.py` | RoBERTa com cabeça de classificação customizada (MLP 768 para 256 para classes, ReLU e dropout 0.3). |

---

## Referência

Sun, C., Huang, L., & Qiu, X. (2019). *Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence*. NAACL 2019. [arXiv:1903.09588](https://arxiv.org/abs/1903.09588)
