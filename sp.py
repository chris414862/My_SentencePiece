from collections import deque
import random as rand
import time


DEBUG = 1
text = "hello_f"#ella_"
vocab = {"h","hel","ell","he","e","ll","f","a_","o_", "h", "l"}


# print func for debugging/logging
def mp(*args,l=1, **kwargs):
    if DEBUG <= l:
        print(*args, **kwargs)

# lattice node
class lattice_node():
    '''
    Lattice node
    '''
    def __init__(self, start:int, end:int, val:str):

        # TODO: ends is list of (1-indexed) indeces indicating 
        self.ends = [end]
        # TODO: starts should be nested lists. 
        self.starts = [start] # This is also the parents end
        self.val = val
        self.children_sets = [set()]

    def __eq__(self, other): #define equivelence
        return  isinstance(other, self.__class__) and self.val == other.val 

    def __hash__(self):
        return hash((self.val, tuple(self.ends)))



# lattice
class my_lattice():
    '''
    Lattice is organized into levels. Each level represents a timestep in which a token from vocab could 
    be generated. Thus len(levels) cannot be greater than the length of the text since single characters
    will always be present in vocab. 

    Inside of each level are lattice_nodes. The lattice_nodes represent tokens that could possibely be 
    generated at that time step. This means len(levels[i]) cannot be greater than len(vocab). For example, 
    with a vocab of {'h','e','l','o','he', 'll','lo, 'hel'} and a 
    given text 'hello', lattice levels will look as follows:
        level 1:
            {'h', 'he', 'hel'}
        level 2:
            {'e', 'l', 'll'}
        level 3:
            {'l', 'o', 'll', 'lo'}
        level 4:
            {'l', 'o'}
        level 5:
            {'o'}

    Each lattice_node, then, can potentially represent a token appearing in different points in the text. For
    example, in level 2 above, 'l' might be generated after 'he', 'hel'. 
    '''
    def __init__(self, vocab, text=None):
        self.root = None
        self.vocab = vocab
        self.text = text
        self.max_len = len(max(vocab, key=len))
        self.levels = None

    def pz(self, z, l=1):
        mp([(z_el.val, z_el.ends) for z_el in z], l=l)


    def pcs(self, z, l=1):
        mp([(i, [(child.val, child.ends) for child in children_set]) for i, children_set in enumerate(z)], l=l)
    # def merge_nodes(self, nodes1, nodes2)
        

    def merge(self, nodes1, nodes2):

    def build(self, text=None):
        self.text = text if not text is None else self.text
        if self.text is None:
            mp("No text given")
            return

        Z = []
        #init root
        self.root = lattice_node(0,0, "")
        self.root.children_sets = self.find_children_sets(self.text, self.root)
        self.pcs(self.root.children_sets)
        z_curr = self.root.children_sets[0]
        i = 0
        while len(z_curr)> 0:
            Z.append(z_curr)
            mp("TIME STEP:", len(Z))
            mp('z_curr:', end='')
            self.pz(z_curr)
            z_new = set()
            for z_curr_el in z_curr:
                
                mp("node:", z_curr_el.val, "starts:", z_curr_el.starts, "ends:", z_curr_el.ends)
                children_sets = self.find_children_sets(self.text, z_curr_el)
                z_curr_el.children_sets = children_sets

                # Create 
                z_new_subset = self.flatten_children_sets(children_sets)
                z_new = self.merge(z_new, z_new_subset)

                mp('children:', end='')
                self.pz(children)

            z_curr = z_new
            mp("len z_new:", len(z_curr))
            break

        self.levels = Z
        return

    def find_children_sets(self, text, node):
        '''
            
        '''
        children_sets = []
        for i in node.ends:
            children = set()
            for j in range(i+1, min(len(text), i+self.max_len)+1):
                # print("checking text:",i,j, text[i:j])

                if text[i:j] in self.vocab:
                    # print("found in vocab")
                    new_val = text[i:j]
                    children.add(lattice_node(i,j, text[i:j]))

            children_sets.append(children)

        return children_sets

    # def sample_tokenization(self):
    #     if self.root is None:
    #         mp("Haven't built lattice yet", l=2)
    #         return []
    #
    #     curr_cands = list(self.root.children)
    #     ret = []
    #     mp("curr_cands", l=2)
    #     self.pz(curr_cands, l=2)
    #     while len(curr_cands) > 0:
    #
    #         lat_node = curr_cands[rand.randint(0,len(curr_cands)-1)]
    #         tok = lat_node.val
    #         mp("chose:", tok, l=2)
    #         ret.append(tok)
    #         curr_cands = list(lat_node.children)
    #         mp("curr_cands", l=2)
    #         self.pz(curr_cands)
    #
    #     return ret

        

            
ml = my_lattice(vocab,text)       
ml.build()
rand.seed(time.time())
# toks = ml.sample_tokenization()
# mp("Random tokenization:", toks, l=3)
        

        


