from my_lattice import my_lattice
import random as rand
import time
import math



#debug vars
DEBUG = 1
# text = "hello_fella_lola_"
# text = "hello_"
text = "hellhelo_"
my_vocab = {
         "a":math.log(.102), "o":math.log(.098), 
         "f":math.log(.081),"h":math.log(.019),
         "l":math.log(.055), "e":math.log(.045),
         "hel":math.log(.054), "ell":math.log(.046), 
         "ll":math.log(.09),"lol":math.log(.01),
         "a_":math.log(.088),"lo_":math.log(.012),
         "o_":math.log(.078),   "he":math.log(.122),
         "lola_":math.log(.048), "ella":math.log(.052),
         "":math.log(1.0)
         }
# my_vocab = {
#          "a":math.log(1.0), "o":math.log(1.0), 
#          "f":math.log(1.0),"h":math.log(1.0),
#          "l":math.log(1.0), "e":math.log(1.0),
#          "hel":math.log(.00001), "ell":math.log(0.01), 
#          "ll":math.log(.1),"lol":math.log(0.01),
#          "a_":math.log(1.0),"lo_":math.log(.00001),
#          "o_":math.log(1.0),   "he":math.log(.1),
#          "lola_":math.log(.0001), "ella":math.log(.0001)
#          }


def logsumexp(log_prob_seq):
    tot = 0
    max_lp = max(log_prob_seq)
    for lp in log_prob_seq:
        # print("lp", lp)
        tot += math.exp(lp - max_lp)
    
    return math.log(tot)+ max_lp


def forward(lattice, vocab, silent=True):
    for level in lattice.levels[1:]:
        for node in level.values():
            node.log_alpha_prob = logsumexp([p.log_alpha_prob  for p in node.parents])+ vocab[node.val]
    
    return logsumexp([n.log_alpha_prob for n in lattice.leaves])


            
def backward(lattice, vocab, log_normalizer, debug=False):

    for level in lattice.levels[::-1]:
        for node in level.values():
            if len(node.children) == 0:
                # node is a leaf
                node.log_beta_prob = 0
            else:
                node.log_beta_prob = logsumexp([c.log_beta_prob + vocab[c.val] for c in node.children]) 

            node.log_marg_prob = node.log_beta_prob + node.log_alpha_prob - log_normalizer
            # if debug:
            #     print("\tfin node", node.val, "node backlp:", node.log_beta_prob, "node forwardlp:", node.log_alpha_prob)
            #     print("\tnode back:", math.exp(node.log_beta_prob), "node forward:", math.exp(node.log_alpha_prob), 
            #             "tot:", math.exp(node.log_alpha_prob)*math.exp(node.log_beta_prob))
            #     print("\tlog_norm", log_normalizer, "norm", math.exp(log_normalizer), "marg:", math.exp(node.log_marg_prob))



def forward_backward(lattice, vocab, debug=False):
    log_normalizer = forward(lattice, vocab)
    backward(lattice, vocab, log_normalizer=log_normalizer, debug=debug)
    return log_normalizer


def update(old_vocab, lattices):
    new_vocab = dict()
    for lattice in lattices:
        # tmp_vocab = dict()
        for level in lattice.levels[1:]:
            for node in level.values():
                # if len(node.children) == 0:
                #     print(node.val)
                new_vocab[node.val] = logsumexp([new_vocab.get(node.val, 0.0), node.log_marg_prob])
                if math.isnan(new_vocab[node.val]):
                    print(lattice)
                    print("-"*20, "node:", node.val)
                    print(node.str()) 
                    sys.exit() 


    # normalize:
    log_normalizer = logsumexp([v for v in new_vocab.values()])
    for k, v in new_vocab.items():
        new_vocab[k] = v - log_normalizer

    # insert remaining values in vocab
    zero_log_prob = -math.inf
    for k, v in old_vocab.items():
        if not k in new_vocab:
            new_vocab[k] = zero_log_prob

    return new_vocab


def lower_bound(lattice, new_vocab, old_vocab):
    log_normalizer = lattice.top.log_marg_prob
    for level in lattice.levels[::-1]:
        for node in level.values():
            if len(node.children) == 0:
                node.lb = math.exp(old_vocab[node.val])* new_vocab[node.val]
            elif len(node.parents) == 0: 
                node.lb = sum([c.lb for c in node.children])
            else: 
                node.lb = math.exp(old_vocab[node.val])*sum([math.exp(c.log_beta_prob)*new_vocab[node.val]+c.lb for c in node.children])

    return lattice.top.lb


def node_sampler(node_set, log_normalizer):
    log_samp_quantile = math.log(rand.uniform(0,1))
    log_accum = None
    curr_node = None
    for node in node_set:
        log_node_prob = node.log_alpha_prob - log_normalizer
        log_accum = logsumexp([log_accum, log_node_prob]) if not log_accum is None else log_node_prob
        if log_samp_quantile < log_accum:
            return node
      
    raise Exception("Logic error occured")

    
def ffbs(lattice, vocab):
    '''
        Forward-filtering, backward-sampling
    '''
    rand.seed(time.time())
    # forward filter
    log_normalizer = forward(lattice, vocab, silent=False)

    sampled_toks = []
    # Sample last tok from a leaf
    leaf = node_sampler(lattice.leaves, log_normalizer)
    assert not leaf is None
    sampled_toks.append(leaf.val)
    
    
    # Sample from parents until top is found
    curr_node = leaf
    while not lattice.top in curr_node.parents:
        # Sample next tok from parents
        conditional_log_normalizer = logsumexp([p.log_alpha_prob for p in curr_node.parents])
        curr_node = node_sampler(curr_node.parents, conditional_log_normalizer)
        sampled_toks.append(curr_node.val)

    return sampled_toks[::-1]


    

if __name__ == "__main__":            

    ml = my_lattice(set(my_vocab.keys()),text)       
    ml.build()

    def report(lattice, lb, step:str, silent=False):
        if not silent:
            print(f"LB after {step}:", lb/math.exp(lattice.top.log_marg_prob))
            print(f"LB after {step} (w/normalizer):", lb/math.exp(lattice.top.log_beta_prob))
            print("log normalizer:", lattice.top.log_beta_prob, "normalizer", math.exp(lattice.top.log_beta_prob))

    print("sampled_toks:", ffbs(ml, my_vocab))
    for i in range(18):
        # e-step
        log_normalizer = forward_backward(ml, my_vocab)
        lb = lower_bound(ml, my_vocab, my_vocab)
        report(ml, lb, "e-step")
        # m-step
        new_vocab = update(my_vocab, ml)
        lb = lower_bound(ml, new_vocab, my_vocab)
        report(ml, lb, "m-step")
        my_vocab = new_vocab
        print("sampled_toks:", ffbs(ml, my_vocab))


