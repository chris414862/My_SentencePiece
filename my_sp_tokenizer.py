import json
from my_lattice import my_lattice
from my_trainer import forward_backward, update, logsumexp, ffbs, lower_bound
import math
from pathlib import Path
from scipy.stats import dirichlet
from kernes_bpe import BytePairEncoder



# Build token vocab using bpe
# Using default from sentencepiece (1,000,000)
SEED_VOCAB_SIZE = 8000
FINAL_VOCAB_SIZE = 6000
DOC_DIR = "./documents"
SAMPLE_TEXT = ""
ALLOWED_CHARS = {
        "'", "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
        }
PUNC = {
        ",",";",",","'",'"',".","?","\\","/","(",")"
        }


doc_dir = Path(DOC_DIR)
for txt_file in doc_dir.iterdir():
    with open(txt_file, encoding='utf-8', errors='ignore') as f:
        doc = ""
        for i, l in enumerate(f.readlines()):
            # if i < 56:
            #     continue
            if i > 4000:
                break
            SAMPLE_TEXT += l.strip().lower()+" "
        # docs.append(doc)

# SAMPLE_TEXT= "In my younger and more vulnerable years my father gave me some advice \
# that I’ve been turning over in my mind ever since.\
# \
# \"Whenever you feel like criticizing anyone,\" he told me, \"just\
# remember that all the people in this world haven’t had the advantages\
# that you’ve had.\"".lower()

# bpe = BytePairEncoder()
# num_merges = SEED_VOCAB_SIZE - 2*len(set(SAMPLE_TEXT))
# bpe.fit(SAMPLE_TEXT, num_merges)
# seed_vocab = set(bpe.tokens.keys())
# print(seed_vocab)

class MySentencePiece():
    def __init__(self):
        pass

    def fit(self, text, final_vocab_size, seed_vocab_size=100000, keep_pct=.9):
        
        self.bpe = BytePairEncoder()
        text = "".join(filter(lambda x: x != "\\", list(text))) 
        self.bpe.fit(text, max(seed_vocab_size - 2*len(set(text)), 2*len(set(text))))
        print("bpe finished", flush=True)
        # print(bpe.tokens)
        #seed vocab
        tot_count = sum([v for v in self.bpe.tokens.values()])
        self.basic_chars = ALLOWED_CHARS.union({c+"_" for c in ALLOWED_CHARS})
        curr_vocab = {k: math.log(v/tot_count) for k, v in self.bpe.tokens.items()}

        # TODO: Think of logical init value for this
        # -inf does not work 
        not_pres = {k:math.log(0.000001) for k in self.basic_chars.difference(set(self.bpe.tokens.keys()))}
        curr_vocab.update(not_pres)
        # sys.exit()
        self.orig_vocab = curr_vocab
        text = self.normalize_text(text)
        while len(curr_vocab) > final_vocab_size:
            lattices = self.build_lattices(text, curr_vocab)
            for i in range(5):
                self.run_e_step(lattices, curr_vocab)
                curr_vocab = self.run_m_step(lattices, curr_vocab)
                print("em round", i+1, "finished", flush=True)

            print("orig len:", len(curr_vocab))
            curr_vocab = self.purge_lowest(curr_vocab, keep_pct, final_vocab_size)
            print("new len:", len(curr_vocab), '\n')

        self.vocab = curr_vocab
            

    def normalize_text(self, text):
        ret = ""
        for w in text.split():
            new_w = "".join(filter(lambda x: x in ALLOWED_CHARS, list(w)))
            if len(new_w) > 0:

                ret += new_w+ "_ "

        return ret

            
    def purge_lowest(self, vocab, keep_pct, final_vocab_size):
        candidates = set(vocab.keys()).difference(self.basic_chars)
        sorted_cands_lst = sorted([(k,vocab[k]) for k in candidates], key=lambda x: x[1], reverse=True)

        cutoff_idx = int((len(vocab)-len(self.basic_chars))*keep_pct)
        purged_cands_lst = sorted_cands_lst[:cutoff_idx]
        new_vocab_len = len(purged_cands_lst) + len(self.basic_chars)
        if new_vocab_len < final_vocab_size:
            diff = final_vocab_size - new_vocab_len 
            purged_cands_lst += sorted_cands_lst[cutoff_idx:cutoff_idx+diff]

        purged_cands = set([k for k,_ in purged_cands_lst])
        ret =  {k:vocab[k] for k in purged_cands.union(self.basic_chars)}
        return ret



    def build_lattices(self, text, vocab): 
        # build lattices for each word in each doc
        lattices = []
        for i, word in enumerate(text.split()):
            ml = my_lattice(vocab, word)
            ml.build()
            lattices.append(ml)

        return lattices

        
    def run_e_step(self, lattices, curr_vocab):
        for lattice in lattices:
            forward_backward(lattice, curr_vocab)


    def run_m_step(self, lattices, curr_vocab):
        # for lattice in lattices:
        new_vocab = update(curr_vocab, lattices)

        return new_vocab

    def get_log_likelihood(self, lattices):
        return sum([l.top.log_beta_prob for l in lattices])

    def get_lower_bound(self, lattices, old_vocab, new_vocab):
        return sum([lower_bound(l, old_vocab, new_vocab) for l in lattices])

    def tokenize(self, text):
        text = self.normalize_text(text)
        lattices = self.build_lattices(text, self.vocab)
        # print("text", text)
        ret =  [ffbs(l, self.vocab) for l in lattices]
        self.run_e_step(lattices, self.vocab)
        print(lattices[0])
        return ret




if __name__ == '__main__':

    msp = MySentencePiece()
    msp.fit(SAMPLE_TEXT, FINAL_VOCAB_SIZE,seed_vocab_size=SEED_VOCAB_SIZE)
    to_toks = "butterfly"
    to_toks = "beautifly"
    print(msp.tokenize(to_toks))
    print(msp.bpe.tokenize(to_toks))



