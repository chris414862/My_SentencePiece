from collections import deque
from typing import List, Set, Dict, Tuple
import math
import random as rand
import time

#debug vars
DEBUG = 1
text = "hello_fella_lola_"
vocab = {
         "a":.102, "o":.098, 
         "f":.091,"h":.009,
         "l":.055, "e":.045,
         "hel":.054, "ell":.046, 
         "ll":.09,"lol":.01,
         "a_":.018,"lo_":.082,
         "o_":.078,   "he":.122,
         "lola_":.048, "ella":.052
         }

# print func for debugging/logging
def mp(*args,l=0, **kwargs):
    if  DEBUG > l:
        print(*args, **kwargs)


class lattice_node():
    '''
        Lattice node. Represents a single token value in a level of my_lattice.
        Token values should be unique within each my_lattice level, but may be instanted
        in several positions in the text. Each unique text position is referred to 
        as an 'instantiation' of the lattice node. 

        Each instantiation will have a different set of children (non-disjoint) pointed to in the next
        lattice level depending on what characters follow the instantiation in the
        text. Similarly, each instantiation will have it's  (different tokens
        from the previous lattice level that could generate this node's token).
        This results in multiple groups of edges being contained in each
        lattice_node; one group for each instance of the token value that can be
        produced in the text at this level in the lattice.

        Note that the same token value with
        the same location in the text may be instantiated at different levels in the 
        lattice. In these cases, the instantiation would have diff

    '''
    def __init__(self,  val:str, end:int):

        self.end_idx = end

        # map from character index in text to instance index in ends/children_sets
        self.val = val #token value
        self.marg_prob = None
        self.alpha_prob = None
        self.beta_prob = None
        self.lb = None
        self.log_marg_prob = None
        self.log_alpha_prob = None
        self.log_beta_prob = None
        self.log_lb = None
        self.children = set()
        self.parents = set()
        self.level_idx = None


    
    def str(self,verbose=False):
        ret = self.__str__()
        if verbose:
            ret += f"\n\t\tlog_alpha: {self.log_alpha_prob}, log_beta: {self.log_beta_prob}, log_marg: {self.log_marg_prob}"
            alpha_prob = math.exp(self.log_alpha_prob) if not self.log_alpha_prob is None else None
            beta_prob = math.exp(self.log_beta_prob) if not self.log_beta_prob is None else None
            marg_prob = math.exp(self.log_marg_prob) if not self.log_marg_prob is None else None
            ret += f"\n\t\talpha: {alpha_prob}, beta: {beta_prob}, marg: {marg_prob}"

        return ret

    def __str__(self):
        ret = f"tok: '{self.val}', parents: {self.parents}, children: {self.children}"

        return ret 


    def __repr__(self):
        return "'"+self.val+"'"

# lattice
class my_lattice():
    '''
        A my_lattice object stores all of the potential tokenizations for a
        text given a fixed vocabulary. These tokenizations are represented
        using a graph resembling a k-partite graph, where k = len(text). Nodes
        in a partition are not connected to eachother. These partitions are
        refered to as levels. The k levels represent time steps in token
        generation for a tokenization model. Each node represents a token that
        can possibly be generated at the i^th time step in the given text. Each
        node is connected to nodes in neighboring levels if the neighboring
        node can likewise be generated before or after the node in the text.
        Therefore, each node in the i^th level is only connected to a subset of
        nodes in the (i-1)^th level and a subset in the (i+1)^th level.  A path
        from the head node  to a leaf represent a unique tokenization of the
        text.


        Inside of each level are lattice_nodes. Each lattice_node
        represents a potential token at a specific position in the text (the
        character index following the token). Thus, a single token may have
        multiple node representations in the i^th level if that token can be
        generated at the i^th time step in multiple positions in the text.


        As a concrete example, with vocab={'h','e','l','o','he', 'el', 'll','lo',
        'hel'} and  text='hello', my_lattice(vocab).build(text) will produce a
        lattice with levels that will look as follows:
            level 1:
                [('h', 1), ('he', 2), ('hel',3)]
            level 2:
                [('e',2), ('l', 3), ('el',3) ('ll',4)]
            level 3:
                [('l', 3), ('l', 4), ('o', 5), ('ll', 4), ('lo', 5)]
            level 4:
                [('l', 4), ('o', 5), ('lo', 5)]
            level 5:
                [('o', 5)]

        Notice that in level 2, the lattice_node representing 'l' and 'el',
        both end at the same index, 3.  Therefore, they will have the same
        edgelist that points to level 3 (in this case {('l', 4), ('lo', 5)}).
        However, since the tokens start at different positions, the edgelists
        pointing to level 1 will differ.

        Also note that in level 3 there are two nodes representing 'l'. This
        separation is necessary not only because the edgelists pointing to
        neighboring levels will be different for each, but also because when
        marginals are computed the probability of the two nodes being generated
        will be distinct since different they will be part of disjoint sets of
        final tokenizations (when represented by paths from head to leaf).


    '''
    def __init__(self, vocab_toks: Set[str], text=None, sep_tok='_'):
        self.top = None
        self.leaves = set()
        self.vocab_toks = vocab_toks
        self.text = text
        self.max_len = len(max(vocab_toks, key=len))
        self.levels = []
        self.starts_at_index_cache = dict()
        self.ends_before_index_cache = dict()
        self.sep_tok = sep_tok

    def pz(self, z, l=1):
        'print level (i.e. latent z variable)'
        mp([(z_el.val, z_el.end_idx) for z_el in z.values()], l=l)


    def merge_next_level_nodes(self, next_level_nodes:Dict[Tuple[str, int], lattice_node], curr_level_node:lattice_node,children:Set[Tuple[str,int]]):
        '''
            Returns next_level_nodes augmented with lattice_nodes representing the tokens in children 
            which have not already been included in next_level_nodes.
            
            Parameters:
                next_level_nodes:  
                        -- Current state of the next level in the lattice.
                children: 
                        -- List of next tokens that can be generated from curr_node.
                        
        '''
        for tok, end_idx in children:
            # token is not already in next_nodes
            if not tuple([tok, end_idx]) in next_level_nodes.keys():
                next_level_node = lattice_node(tok, end_idx)
                next_level_nodes[(tok, end_idx)] = next_level_node
            else:
                next_level_node = next_level_nodes[tuple([tok, end_idx])]

            next_level_node.parents.add(curr_level_node)
            curr_level_node.children.add(next_level_node)




        return next_level_nodes


    def build(self, text:str=None):
        '''
            Build the lattice structure using the given text.
        '''
        self.text = text if not text is None else self.text
        if self.text is None:
            mp("No text given")
            return

        #init root/top
        self.top = lattice_node("", 0)
        self.top.alpha_prob = 1
        self.top.log_alpha_prob = 0 

        # z_curr is dictionary mapping tuple[token, end_idx] --> lattice_nodes
        # represents (posterior) latent z variable at current time step
        z_curr = {("",0):self.top}
        # self.pz(z_curr)

        while len(z_curr)> 0:
            assert isinstance(z_curr, dict), "latent variable z is stored as a dict"
            self.levels.append(z_curr)

            # build next level 
            z_next = dict()
            
            # Find all possible tokens for next level
            for curr_node in z_curr.values():
                # mp("tok:", curr_node.val, "end_idx:", curr_node.end_idx)

                # get set of tokens that can be generated after curr_node's token 
                children = self.find_children(curr_node.end_idx)
                if len(children) == 0:
                    # it's a leaf
                    curr_node.beta_prob = 1
                    curr_node.log_beta_prob = 0
                    self.leaves.add(curr_node)

                # Merge children into next level's node set 
                z_new = self.merge_next_level_nodes(z_next, curr_node, children)
                curr_node.level_idx = len(self.levels)-1

            z_curr = z_next
            # mp("len z_new:", len(z_curr))

        return

    def find_children(self, i:int):
        '''
            Returns a set(tuple(token, end_idx)) object that indicates the
            tokens that can be generated from text[i] using the tokens in self.vocab. 

            Checks progressively longer strings, starting from self.text[i], for
            membership in self.vocab.

            Results are cached to prevent unnecessary calulations.

            Parameters:
                i: int
                    -- index to start child search from 
        '''
         
        if i in self.starts_at_index_cache:
            return self.starts_at_index_cache[i]

        children = set()
        for j in range(i+1, min(len(self.text), i+self.max_len)+1):

            # don't add token if nect character is '_'
            if len(self.text) > j  and self.text[j] == self.sep_tok:
                continue

            elif self.text[i:j] in self.vocab_toks:
                children.add(tuple([self.text[i:j], j]))

        self.starts_at_index_cache[i] = children
        return children


    def rand_sample_tokenization(self):
        if self.top is None:
            mp("Haven't built lattice yet", l=2)
            return []

        ret = []
        curr_cands = self.top.children
        curr_text_idx = 0
        while len(curr_cands) > 0:
            rand_idx = rand.randint(0,len(curr_cands)-1)
            node = list(curr_cands)[rand_idx]
            tok = node.val

            # Error check: if current sequece of tokens is possible to generate
            # then the last index of it's concatenated string should be one of
            # the instance ending indeces in the current node 
            curr_text_idx += len(tok)
            assert curr_text_idx == node.end_idx

            ret.append(tok)

            curr_cands = node.children


        return ret

        
    def random_sample_tokenization(self):
        if self.top is None:
            mp("Haven't built lattice yet", l=2)
            return []

        ret = []
        curr_cands = self.top.children
        curr_text_idx = 0
        while len(curr_cands) > 0:
            rand_idx = rand.randint(0,len(curr_cands)-1)
            node = list(curr_cands)[rand_idx]
            tok = node.val

            # Error check: if current sequece of tokens is possible to generate
            # then the last index of it's concatenated string should be one of
            # the instance ending indeces in the current node 
            curr_text_idx += len(tok)
            assert curr_text_idx == node.end_idx

            ret.append(tok)
            # mp('token#', len(ret), "chose:", tok, "curr_text_idx", curr_text_idx,  "level_idx", node.level_idx, l=2)

            curr_cands = node.children
            # mp("ret", ret, l=2)
            # mp("curr_cands", l=2)
            # mp([node.val for node in curr_cands])


        return ret

    #TODO: Write viterbi algo

    def __str__(self):
        ret = "LATTICE TEXT: "+self.text+"\n"
        for i in range(len(self.levels)):
            ret += f"LEVEL: {i}\n"
            for node in self.levels[i].values():
                ret += "\t"+node.str(verbose=DEBUG>0)+"\n"

        return ret



if __name__ == "__main__":            
    ml = my_lattice(set(vocab.keys()),text)       
    ml.build()
    rand.seed(time.time())
    toks = ml.random_sample_tokenization()
    print("Random tokenization:", toks)
        

        


