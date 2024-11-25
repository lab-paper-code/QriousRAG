# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=g-explicit-length-test,dangerous-default-value,redefined-outer-name,g-explicit-bool-comparison, missing-function-docstring,missing-module-docstring,raise-missing-from,g-complex-comprehension
import argparse
import collections
import json
import re
import string

import nltk
import numpy as np
from rouge_score import rouge_scorer
from rouge_score import scoring

# nltk.download('punkt')
from transformers import pipeline
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# answer normalization
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


# def _rouge_calculation(candidates,
#                         references,
#                         metrics=['rougeLsum']):
#     """Internal function for rouge scoring.

#     If two references are provided,
#     the best score is chosen for each instance.

#     Args:
#         candidate: list of predicted long answers
#         references: list of list of references to score hypotheses against
#         metrics: evaluation metric

#     Returns:
#         dictionary representation of rouge scores
#     """
    
#     scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
#     aggregator = scoring.BootstrapAggregator()

#     for i in range(len(candidates)):
#         best_score = []
#         for j in range(len(references[i])):
#             score = scorer.score(references[i][j], candidates[i])
#             if j == 0:
#               best_score.append(score)
#             elif score['rougeLsum'].fmeasure > best_score[0]['rougeLsum'].fmeasure:
#                 best_score.insert(0, best_score)
#         aggregator.add_scores(best_score[0])

#     scores = {m: [] for m in metrics}

#     for m in metrics:
#         fmeasure = aggregator.aggregate()[m].mid.fmeasure
#         scores[m].append(fmeasure)

#     for m in scores:
#         scores[m] = 100 * sum(scores[m]) / len(scores[m])

#     return scores


def rouge(candidates,
          references,
          metrics=['rougeLsum']):
  """Main function for rouge scoring.

  If two references are provided,
  the best score is chosen for each instance.

  Args:
    candidats: list of generated answs
    references: list of list of long answs
    metrics: list of evaluation metrics

  Returns:
    dictionary representation of rouge scores
  """
  if 'rougeLsum' in metrics:
    candidates = ['\n'.join(nltk.sent_tokenize(text.lower())) for text in candidates]
    references = [['\n'.join(nltk.sent_tokenize(text.lower())) for text in reference] for reference in references]
  # print(candidates)
  # print(references)
  scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
  aggregator = scoring.BootstrapAggregator()

  for i in range(len(candidates)):
      best_score = []
      for j in range(len(references[i])):
          score = scorer.score(references[i][j], candidates[i])
          if j == 0:
            best_score.append(score)
          elif score[metrics[0]].fmeasure > best_score[0][metrics[0]].fmeasure:
              best_score.insert(0, score)
      aggregator.add_scores(best_score[0])

  scores = {m: [] for m in metrics}

  for m in metrics:
      fmeasure = aggregator.aggregate()[m].mid.fmeasure
      scores[m].append(fmeasure)

  for m in scores:
      scores[m] = 100 * sum(scores[m]) / len(scores[m])

  return scores


# def _exact_presence(short_answers, n_context):
#   """Verify if any of the answers is present in the given context.

#   Args:
#     short_answers: list of short answers to look for in the context
#     context: a paragraph to search for short answers

#   Returns:
#     true if any of the short answers is present in the context
#   """

#   n_short_answers = [normalize_answer(sa) for sa in short_answers]

#   for ans in n_short_answers:
#     if ans in n_context:
#       return True

#   return False


def str_em(predictions, short_answers):
  """Compute STR-EM metric.

  Args:
    predictions: list of predicted answers
    short_answers: list of list of short answers that should be present in the pred_answer

  Returns:
    Value of the STR-EM metric
  """
  acc = []

  for context, short_answ in zip(predictions, short_answers):
    loc_acc = []
    n_context = normalize_answer(context)
    
    for answs in short_answ:
      n_short_answers = [normalize_answer(sa) for sa in answs]
      # print(n_short_answers)
      loc_acc.append(any(n_answ in n_context for n_answ in n_short_answers))
    # print(loc_acc)
    acc.append(np.mean(loc_acc))

  return 100 * np.mean(acc)


def compute_len(predictions, target_keys=None):
  """Compute average lenght of predictions.

  Args:
    predictions: list of predicted answers

  Returns:
    average length of predicted answers
  """

  res = 0
  cntr = 0

  for pred in predictions:
    res += len(pred.split())
    cntr += 1

  return res / cntr


def _get_tokens(s):
  """Split the string into tokens.

  Args:
    s: string to be split

  Returns:
    list of tokens
  """

  if not s:
    return []
  return normalize_answer(s).split()


def _compute_f1(a_gold, a_pred):
  """Compute F1 score between two strings.

  Args:
    a_gold: string one
    a_pred: string two

  Returns:
        f1 score
  """

  gold_toks = _get_tokens(a_gold)
  pred_toks = _get_tokens(a_pred)

  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())

  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)

  if num_same == 0:
    return 0

  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)

  return f1


def _compute_exact(a_gold, a_pred):
  """Check whether two strings are equal up to normalization.

  Args:
    a_gold: string one
    a_pred: string two

  Returns:
    1 if two strings are equal up to normalization and 0 otherwise
  """
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def score_qa_accuracy(predictions, asqa, target_keys=None):
  """Compute QA metrics.

  Args:
    predictions: {qa_key: short_answer} output of ROBERTA predictions of short
      answers to disambiguated questions
    asqa: dict representation of the asqa dataset
    target_keys: an optional set of keys. If provided, only keys from this set
      are used for evaluation

  Returns:
    QA metrics (QA-EM, QA-F1, QA-Hit)
  """

  em, f1, bins = [], [], []

  for key, instance in asqa.items():

    if target_keys is not None and key not in target_keys:
      continue

    loc_counter, loc_em, loc_f1 = 0, 0, 0

    for idx, qa_pair in enumerate(instance['qa_pairs']):
      answers = qa_pair['short_answers']
      full_key = key + '_' + str(idx)
      prediction = predictions[full_key]

      if not isinstance(prediction, list):
        prediction = [prediction]

      loc_em += max([_compute_exact(a, p) for a in answers for p in prediction])
      loc_f1 += max([_compute_f1(a, p) for a in answers for p in prediction])

      loc_counter += 1

    em.append(loc_em / loc_counter)
    f1.append(loc_f1 / loc_counter)
    bins.append(loc_em == loc_counter)

  return {
      'QA-EM': 100 * np.mean(em),
      'QA-F1': 100 * np.mean(f1),
      'QA-Hit': 100 * np.mean(bins)
  }

def disambig(context,asqa):
  print(asqa)
  context=context[0]
  question=asqa['question']
  follow=eval(asqa['follow_up_questions'])
  short=eval(asqa['short_answers'])

  print(context)
  print(question)
  print(follow)
  print(short)
  
  model_name = "deepset/roberta-base-squad2"
  nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)
  cnt=0
  loc_f1=0
  print(len(follow))
  for i in range(len(follow)):
      QA_input = {
          'question': follow[i],
          'context': context
      }
      print(follow[i])
      res = nlp(QA_input)
      prediction=[]
      prediction.append(res['answer'])
      print(res)
      print(len(short))
      ans=[]
      for a in short[i]:
          for p in prediction:
              res=_compute_f1(a, p)
              ans.append(res)
      loc_f1+=max(ans)
      cnt+=1
  f1=loc_f1/cnt
  return 100 * f1


def evaluate(candidates, asqa,
                       rouge_metrics=['rougeLsum']):
    """This function computes values of the following metrics: LENGTH, ROUGE-L, STR-EM, QA-EM, QA-F1, QA-HIT, OVERALL SCORE.

    Args:
        candidates: a list of generated answers
        asqa: list of dicts representation of the ASQA dataset
        rouge_metrics: list of ROUGE metrics to compute

    Returns:
        dict with scores: {rougeLsum: ..., str_em: ..., disambig_f1: ..., DR: ...}
    """
    references = []
    short_answers =  []
    follow=[]
    for sample in asqa:
        references.append(eval(sample['long_answers']))
        short_answers.append(eval(sample['short_answers']))
    # calculate rouge score       
    scores = rouge(
        candidates,
        references,
        metrics=rouge_metrics)
    scores['length'] = compute_len(candidates)
    scores['str_em'] = str_em(candidates, short_answers)
    scores['Disambig-F1']=disambig(candidates,asqa[0])
    # if 'qa' in hypotheses:
    #     qa_scores = score_qa_accuracy(hypotheses['qa'], asqa, target_keys)

    #     for m in qa_scores:
    #     scores[m] = qa_scores[m]

    #     if 'rougeLsum' not in scores:
    #     scores['ovscore'] = 'Undefined'
    #     else:
    scores['ovscore'] = np.sqrt(scores['Disambig-F1'] * scores['rougeLsum'])

    return scores


# def parse_args(argv=None):
#   parse = argparse.ArgumentParser()
#   parse.add_argument('--asqa', type=str, help='Path to the ASQA data')

#   parse.add_argument(
#       '--split',
#       type=str,
#       default='dev',
#       help='What data split you want to evaluate on')

#   parse.add_argument(
#       '--predictions', type=str, help='Path to model predictions')
#   parse.add_argument(
#       '--roberta_output', type=str, help='Path to Roberta output')
#   parse.add_argument('--out_dir', type=str, help='Output path')
#   # parse the arguments
#   return parse.parse_args(argv)

# if __name__ == '__main__':
#   args = parse_args()
#   try:
#     with open(args.asqa, 'r') as handler:
#       asqa = json.load(handler)[args.split]
#   except FileNotFoundError:
#     raise ValueError('Cannot open ASQA, abort')
#   except KeyError:
#     raise ValueError('Wrong split is provided, abort')

#   try:
#     with open(args.predictions, 'r') as handler:
#       predictions = json.load(handler)
#   except FileNotFoundError:
#     raise ValueError('Cannot open predictions, abort')

#   hypotheses = {'answers': predictions}

#   if args.roberta_output is not None:
#     try:
#       with open(args.roberta_output, 'r') as handler:
#         qa_preds = json.load(handler)
#     except FileNotFoundError:
#       raise ValueError('Cannot open predictions, abort')

#     hypotheses['qa'] = qa_preds

#   scores = compute_all_scores(hypotheses, asqa)
#   print(json.dumps(scores, indent=2))
#   out_fn = args.out_dir + '/final_eval_results.json'
#   with open(out_fn, 'w') as outfile:
#     json.dump(scores, outfile)