import json, os
import spacy
#import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
#nlp = en_core_web_md.load()

interrogative_words = ['what', 'who', 'which', 'whom', 'where', 'when', 'how', 'whose', 'why']
THRESHOLD = 0.5
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wikipedia.json')
with open(path) as f:
  wiki = json.load(f)

def get_answer(question_str):
  print(question_str)
  question_str = question_str.lower()
  question = nlp(question_str)

  spans = []
  for i in range(len(question)):
    token = question[i]
    if token.dep_ == 'compound':
      span = question[question[token.head.i].left_edge.i : question[token.head.i].i + 1]#.right_edge.i + 1]
      print(span)
      spans.append(span)

  for span in spans:
    with question.retokenize() as retokenizer:
        retokenizer.merge(span)

  tokens = [token for token in question if not nlp.vocab[token.text].is_stop and not token.text == 'cal poly']

  wh_word = [token.text for token in question if token.text.lower() in interrogative_words][0]

  return check_sidebox(tokens, wiki['sidebox'])

def check_sidebox(keywords, sidebox):
  scores = {key: comp_kws_and_phrase(keywords, key.lower()) for key in sidebox}
  best_k = max(sidebox, key=lambda k: scores[k])
  if scores[best_k] > THRESHOLD:
    return sidebox[best_k]
  else:
    print(best_k, scores[best_k])

def comp_kws_and_phrase(keywords, phrase_str):
  phrase = nlp(phrase_str)
  scores = []
  for kw in keywords:
    #scores.append(max(kw.similarity(x) for x in phrase))
    scores.append(kw.similarity(phrase))
  return sum(scores) / len(scores)
