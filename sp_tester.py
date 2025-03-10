import json
from my_lattice import my_lattice
from my_trainer import forward_backward, update, logsumexp, ffbs, lower_bound
import math
from pathlib import Path
from scipy.stats import dirichlet


DOC_DIR = './documents'
# VOCAB_FILE = './vocabs/test_vocab.json'
#
#
# with open(VOCAB_FILE) as f:
#     my_vocab = json.load(f)

my_vocab = {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m", 
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        }
punc = {

        ",",
        ";",
        ",",
        "'",
        '"',
        ".",
        "?",
        "\\", 
        "/",
        "(",

        ")"
        }
# my_vocab = {
#          "a":math.log(.102), 
#          "o":math.log(.098), 
#          "f":math.log(.081),
#          "h":math.log(.019),
#          "l":math.log(.055), 
#          "e":math.log(.045),
#          "hel":math.log(.054), "ell":math.log(.046), 
#          "ll":math.log(.09),"lol":math.log(.01),
#          "a_":math.log(.088),"lo_":math.log(.012),
#          "o_":math.log(.078),   "he":math.log(.122),
#          "ola_":math.log(.048), 
#          "lla_":math.log(.052),
#          "":math.log(1.0)
#          }
new_vocab = set()
for t in my_vocab:
    for t2 in my_vocab.difference({t}):
        new_vocab.add(t)
        new_vocab.add(t+t2)
my_vocab = new_vocab
new_vocab = set()
for t in my_vocab:
    new_vocab.add(t)
    new_vocab.add(t+"_")

my_vocab = new_vocab
my_vocab =  my_vocab.union(punc)
init_weights = dirichlet.rvs([1.0]*len(my_vocab))[0]
my_vocab= {tok:math.log(iw) for iw, tok in zip(init_weights, my_vocab)}

# print(my_vocab)
# sys.exit()


docs = [
# "hello",
# "fella", 
# "lola"
]

doc_dir = Path(DOC_DIR)
for txt_file in doc_dir.iterdir():
    with open(txt_file, encoding='utf-8', errors='ignore') as f:
        doc = ""
        for i, l in enumerate(f.readlines()):
            if i < 56:
                continue
            elif i > 65:
                break
            doc += l.strip()+" "
        docs.append(doc)



#preprocessing
mv_toks = set(my_vocab.keys())
for i in range(len(docs)):
    split_toks = docs[i].split()
    for j in range(len(split_toks)):
        split_toks[j] = split_toks[j].lower()

        new_tok = ""
        for c in split_toks[j]:
            if c in mv_toks:
                new_tok += c

        if new_tok[-1] != '_':
            new_tok += '_ '

        split_toks[j] = new_tok


            
    docs[i] = "".join(split_toks)
    

# build lattices for each word in each doc
lattices = []
for doc in docs:
    # ml = my_lattice(my_vocab, doc)
    # ml.build()
    # lattices.append(ml)
    for i, word in enumerate(doc.split()):
        ml = my_lattice(my_vocab, word)
        ml.build()
        lattices.append(ml)

print([l.text for l in lattices])
# sys.exit()

def run_e_step(lattices, curr_vocab):
    for lattice in lattices:
        forward_backward(lattice, curr_vocab)
    

def run_m_step(lattices, curr_vocab):
    # for lattice in lattices:
    new_vocab = update(curr_vocab, lattices)

    return new_vocab

def get_log_likelihood(lattices):
    return sum([l.top.log_beta_prob for l in lattices])

def get_lower_bound(lattices, old_vocab, new_vocab):
    return sum([lower_bound(l, old_vocab, new_vocab) for l in lattices])

# EM to maximize likelihood
num_runs = 18
for i in range(num_runs):
    run_e_step(lattices, my_vocab)
    new_vocab = run_m_step(lattices, my_vocab)
    print(f"ll after run {i+1}: {get_log_likelihood(lattices)}")
    my_vocab = new_vocab








        





