import json
import os
import copy
import spacy
import en_core_web_md
from spacy.lang.en.stop_words import STOP_WORDS
import torch
# from models import InferSent
from .POSTree import POSTree
import nltk
import numpy as np
from spacy.matcher import Matcher
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from spacy.tokens import Doc
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
from spacy.tokens import Token
from nltk.wsd import lesk
from nltk.stem.porter import *
import pandas as pd
stemmer = PorterStemmer()

nlp = en_core_web_md.load()

nltk.download('punkt')

WH_WORDS = {
    'what': {},
    'who': {'PERSON', 'NORP', 'ORG'},
    'which': {},
    'whom': {'PERSON'},
    'where': {'FAC', 'GPE', 'LOC'},
    'when': {'DATE', 'TIME'},
    'whose': {'PERSON'},
    'how many': {'QUANTITY'},
    'how much': {'PERCENT', 'MONEY', 'QUANTITY'},
}

THRESHOLD = 0.8

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wikipedia.json')
with open(path) as f:
  wiki = json.load(f)


class WSDPipeline(object):
    def __init__(self, nlp):
        Token.set_extension('sense', default=None, force=True)

    def __call__(self, doc):
        # new_doc = span_compound(doc)

        sent = ' '.join([token.text for token in doc])

        for token in doc:
            # Wordnet only uses a single letter 'n','v','a','r' for tags, so we have to translate
            if token.tag_.lower()[0] == 'j':
                tag = 'a'
            else:
                tag = token.tag_.lower()[0]

            if tag not in "nvar":
                continue

            sense = lesk(sent, token.lemma_, tag)
            if sense:
                token._.set('sense', sense)

        return doc


nlp.add_pipe(WSDPipeline(nlp), name='wn_wsd')
print("current pipelines: ", nlp.pipeline, "\nstages:", ", ".join(stage[0] for stage in nlp.pipeline))

def span_compound(doc):
    spans = []
    i = 0

    while (i < len(doc)):
        token = doc[i]
        if token.dep_ == 'compound':
            span = doc[i : token.head.i + 1]
            i = token.head.i + 1
            spans.append(span)
        else:
            i += 1

    for span in spans:
        with doc.retokenize() as retokenizer:
            retokenizer.merge(span)
    return doc

def tokenize(doc):
    return [token for token in doc]

def remove_stop_words(tokens):
    tokens = [token for token in tokens if not token.is_stop and token.norm_ != 'cal poly' and token.pos_ != 'PUNCT']
    return tokens

def preprocess(text):
    doc = nlp(text)
    doc = span_compound(doc)
    tokens = tokenize(doc)
    tokens = remove_stop_words(tokens)
    return tokens

def get_wh_word(question):
    for wh_word in WH_WORDS:
        if question[:len(wh_word)].lower() == wh_word:
            return wh_word

def retokenize_ents(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    return doc

def check_infobox(keywords, infobox):
  scores = {key: comp_kws_and_phrase(keywords, key.lower()) for key in infobox}
  best_k = max(infobox, key=lambda k: scores[k])
  if scores[best_k] > THRESHOLD:
    return infobox[best_k]

def comp_kws_and_phrase(keywords, phrase_str):
  phrase = nlp(phrase_str)
  scores = []
  for kw in keywords:
    scores.append(kw.similarity(phrase))
  return sum(scores) / len(scores)

def get_sentence_section(wiki):
  data = {'sentences': [], 'sections': []}

  for section in wiki['Sections']:
    title = section['Title']
    sentences = nltk.sent_tokenize(section['Information'])
    for sentence in sentences:
      data['sentences'].append(sentence)
      data['sections'].append(title)
    if len(section['Subsections']) != 0:
      for subsection in section['Subsections']:
        sub_title = subsection['Title']
        sub_sentences = nltk.sent_tokenize(subsection['Information'])
        for sub_sentence in sub_sentences:
          data['sentences'].append(sub_sentence)
          data['sections'].append(title + ',' + sub_title)
        if len(subsection['Subsections']) != 0:
          for sub_subsection in subsection['Subsections']:
            sub_subtitle = sub_subsection['Title']
            sub_subsentences = nltk.sent_tokenize(sub_subsection['Information'])
            for sub_subsentence in sub_subsentences:
              data['sentences'].append(sub_subsentence)
              data['sections'].append(title + ',' + sub_title + ',' + sub_subtitle)
  return data

data = get_sentence_section(wiki)
data_df = pd.DataFrame(data = data)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentence_section.csv')

if not os.path.exists(path):
  data = get_sentence_section(wiki)
  data_df = pd.DataFrame(data = data)
  data_df.to_csv(path)
else:
  data_df = pd.read_csv(path)

sections = data_df['sections']
sentences = data_df['sentences'].str.lower()


# max similarity algorithm
def get_semantic_vector(synsets_1, synsets_2):
    vector_length = max([len(synsets_1), len(synsets_2)])
    semantic_vector = [0.0] * vector_length
    count = 0

    if len(synsets_2) == 0:
        return semantic_vector, count
    for i in range(len(synsets_1)):
        synset_1 = synsets_1[i][1]
        max_similarity = max(
            [synset_1.wup_similarity(synset_2[1]) if synset_1.wup_similarity(synset_2[1]) else 0 for synset_2 in
             synsets_2])
        semantic_vector[i] = max_similarity
        if max_similarity > THRESHOLD:
            count += 1
    return semantic_vector, count


def get_semantic_similarity(semantic_vector_s1, count_1, semantic_vector_s2, count_2):
    S = np.dot(semantic_vector_s1, semantic_vector_s2)
    denominator = sum([count_1, count_2]) / 1.8

    if sum([count_1, count_2]) == 0:
        denominator = len(semantic_vector_s1) / 2
    if denominator == 0:
        return 0
    return S / denominator


def get_section_semantic(question_tokens, questions, sections):
    best_score = 0
    most_similar_question = ''
    section = ''
    for i in range(len(questions)):
        other_question_doc = nlp(questions[i])

        # remove stop words
        other_question_tokens = remove_stop_words(other_question_doc)
        # other_question_tokens = [token for token in other_question_doc if token.tag_.startswith('N') or token.tag_.startswith('V')]

        synsets_1 = [(token, token._.sense) for token in question_tokens if token._.sense]
        synsets_2 = [(token, token._.sense) for token in other_question_tokens if token._.sense]

        if len(synsets_1) >= len(synsets_2):
            semantic_vector_1, count_1 = get_semantic_vector(synsets_1, synsets_2)
            semantic_vector_2, count_2 = get_semantic_vector(synsets_2, synsets_1)
        else:
            semantic_vector_1, count_1 = get_semantic_vector(synsets_2, synsets_1)
            semantic_vector_2, count_2 = get_semantic_vector(synsets_1, synsets_2)

        similarity = get_semantic_similarity(semantic_vector_1, count_1, semantic_vector_2, count_2)

        if similarity > best_score:
            best_score = similarity
            most_similar_question = questions[i]
            section = sections[i]
    print(best_score)
    print(most_similar_question)

    if best_score > THRESHOLD:
        return section
    else:
        return None

def get_section_tfidf(question, sections):
    tokenizer = lambda text: [token.norm_ for token in preprocess(text)]
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)

    X = vectorizer.fit_transform(sections)
    y = list(range(len(sections)))
    classifier = KNeighborsClassifier(n_neighbors=1, metric=cosine_similarity)
    classifier.fit(X, y)

    question_X = vectorizer.transform([question])
    question_y = classifier.predict(question_X)
    return sections[question_y[0]]

def get_section(question_tokens, sentences, sections):
  semantic_section = get_section_semantic(question_tokens, sentences, sections)
  print(semantic_section)
  question = ' '.join(token.text for token in question_tokens)
  lexical_section = get_section_tfidf(question, sections)
  print(lexical_section)
  if semantic_section:
    section = semantic_section
  elif lexical_section:
    section = lexical_section
  else:
    section = None
  return section

def convert_tokens_to_doc(tokens, nlp):
  str_tokens = [token.text.lower() for token in tokens]
  spaces = [True] * len(str_tokens)
  doc = Doc(nlp.vocab, words=str_tokens, spaces=spaces)
  return doc

def get_sent_avg_embedding(question_tokens, section):
  quest_doc = convert_tokens_to_doc(question_tokens, nlp)

  max_score = 0
  most_similar_sent = ''

  for sent in section:
    # span compound
    sent_tokens = preprocess(sent)
    sent_doc = convert_tokens_to_doc(sent_tokens, nlp)
    score = quest_doc.similarity(sent_doc)
    print(score)
    if score > max_score:
      max_score = score
      most_similar_sent = sent

  return most_similar_sent


def get_stanford_parse(question):
    with open('temp_in.txt', 'w') as f:
        f.write(question + '?')

    result = os.system('sh /usr/local/Cellar/stanford-parser/3.9.2_1/libexec/lexparser.sh temp_in.txt > temp_out.txt')

    with open('temp_out.txt') as f:
        result = f.read().split('\n\n')[0]

    return result


def pattern_match(question, sent):
    print("sent", sent)
    nlp_sent = nlp(sent)
    retok_sent = retokenize_ents(nlp(sent))

    tree = POSTree(get_stanford_parse(question))
    reformulated_query_text = tree.adjust_order()
    reformulated_query = reformulated_query_text.split()
    print(reformulated_query)

    matcher = Matcher(nlp.vocab)
    pattern = [{'LOWER': x} if x != '**blank**' else {} for x in reformulated_query]
    print(pattern)
    blank_index = reformulated_query.index('**blank**')
    matcher.add('main', None, pattern)

    matches = []
    for _, start, end in matcher(nlp_sent):
        print(nlp_sent[start:end])
        for tok in retok_sent:
            if nlp_sent[start + blank_index].text in tok.text:
                matches.append(tok)
    return reformulated_query_text.replace('**blank**', matches[0].text) if len(matches) > 0 else None


def get_matching_type_ents(question, sent):
    wh_word = get_wh_word(question)
    answer_types = WH_WORDS[wh_word]
    if len(answer_types) == 0:
        return list(sent.ents)

    return [ent for ent in sent.ents if ent.label_ in answer_types]

def get_surrounding_words(doc, ents, question_keywords, question):
  score_dict = {}
  question_keywords = [str(tok) for tok in question_keywords]
  for ent in ents:
    score = 1
    for tok in doc:
      if tok.text == ent.text:
        cur = tok
        break
    score_dict[ent.text] = 0
    while cur.head != cur:
      if cur.head.text in question_keywords:
        score_dict[ent.text] += score
      cur = cur.head
      score /= 2.0
  return max(score_dict, key=lambda x: score_dict[x])


def recognize_entity(answer, nlp, entity):
    entitites = []
    answer_doc = nlp(answer)
    for ent in answer_doc.ents:
        if ent.label_ == entity:
            entitites.append(ent.text)
    return ', '.join(entitites)


def create_response(answer, question_doc):
    tree = POSTree(get_stanford_parse(question_doc))
    reformulated_query = tree.adjust_order()
    return reformulated_query.replace('**blank**', answer)


def get_answer(question):
    question_tokens = preprocess(question)
    sidebox_answer = check_infobox(question_tokens, wiki['sidebox'])
    if sidebox_answer:
        return create_response(sidebox_answer, question)

    section = get_section(question_tokens, sentences, sections)
    if section is None:
        return "Can't seem to find the answer, sorry about that."

    section_text = ' '.join(data_df.loc[sections==section]['sentences'])
    print("section text", section_text)
    section = nltk.sent_tokenize(section_text)

    sent = get_sent_avg_embedding(question_tokens, section)
    match = pattern_match(question, sent)

    if match:
        print("******   PATTERN MATCH ANSWER   ******")
        return match

    ents = get_matching_type_ents(question, nlp(sent))
    doc = retokenize_ents(nlp(sent))
    answer = get_surrounding_words(doc, ents, question_tokens, question)
    print("******   SURROUNDING_WORDS ANSWER   ******")

    return create_response(answer, question)
