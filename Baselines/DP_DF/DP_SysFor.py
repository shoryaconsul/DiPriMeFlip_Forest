from collections import Counter, defaultdict
from . import DP_Tree

class DP_SysFor:
    _forest = []
    _previous_roots = []

    def __init__(self, records=None, target_attribute_index=0, trace=0, 
                 num_trees=4, max_height=5, min_support=0, epsilon_budget=1.0,
                 attribute_names=None, pruning=False):
        self._forest = []
        self._previous_roots = []
        
        epsilon_per_query = epsilon_budget / (num_trees*(2*max_height - 1.)) # 2 queries per node, except the last level
        #print( "epsilon_per_query = "+str(epsilon_per_query) )

        for t in range(num_trees):
            self._forest.append( self._each_tree(records[:], target_attribute_index, trace, max_height, min_support, attribute_names, epsilon_per_query, pruning) )
#            print('Tree ',t)



    def _each_tree(self, records, target_attribute_index, trace, max_height, min_support, attribute_names, epsilon_per_query, pruning):
        #print( "Starting Tree Building..." )
        diff_priv_tree = DP_Tree.DP_Tree(instances=records, target_attribute_index=target_attribute_index, trace=trace, 
                                max_height=max_height, min_support=min_support, epsilon_budget=None,
                                attribute_names=attribute_names, epsilon_per_query=epsilon_per_query, previous_roots=self._previous_roots)
        
        self._previous_roots.append( diff_priv_tree._root_node._attribute_index )

        if pruning:
            num_prunings = diff_priv_tree.prune_tree()
            #print("num_pruning = "+str(num_prunings))
        #printed_tree = diff_priv_tree.print_tree()
        #rule_list_text = diff_priv_tree.print_rule_list()
        #print( "Finished Tree Building!" )
        return diff_priv_tree #{'tree':diff_priv_tree._root_node, 'printed_tree':printed_tree, 'rule_list':rule_list_text}


    # a method intended to be "protected" that can implement the recursive algorithm to classify an instance given a tree
    def _classify(self, node, record):
        if not node._children: # if leaf
            #print("rec[class]={} & prediction={}".format(instance[0], node._default_prediction))
            return {'class':Counter(node._class_counts).most_common(1)[0][0], 'conf':node._confidence}
        else:
            #print(',', end='')
            attr = node._attribute_index
            rec_val = record[attr]
            child = None
            for i in node._children:
                if i._parent_value == rec_val:
                    child = i
                    #print("attr={}, rec_val={}, child={}".format(attr, rec_val, child._class_counts))
                    break
            if child==None:
                return {'class':Counter(node._class_counts).most_common(1)[0][0], 'conf':node._confidence}
            return self._classify(child, record)


    def classify_list(self, test_data):
        predicted_labels = []
        for record in test_data:
            votes = defaultdict(float)
            for tree in self._forest:
                prediction = self._classify(tree._root_node, record)
                votes[prediction['class']] += prediction['conf']
            #print( "votes: "+str(votes) )
            predicted_labels.append( Counter(votes).most_common(1)[0][0] ) # the predicted label is the label with the most confidence over all trees.
#            print('# PREDICTIONS: ',len(predicted_labels))
        return predicted_labels


    def evaluate_accuracy(self, test_data, target_index=0):
        predicted_labels = self.classify_list(test_data)
        actual_labels = [x[target_index] for x in test_data]
        counts = Counter([x == y for x, y in zip(predicted_labels, actual_labels)])
        return counts[True], counts[False], float(counts[True]) / len(test_data)


    def get_num_prunings(self):
        total = 0
        for tree in self._forest:
            total += tree._num_prunings
        return total

    
    def get_rules_per_level(self):
        rules_per_level = defaultdict(int)
        av_conf_per_level = defaultdict(float)
        for tree in self._forest:
            rules, conf = tree.rules_and_conf_per_level()

            for k,v in rules.items(): # each k is a level
                rules_per_level[k] += v
                av_conf_per_level[k] += conf[k]/len(self._forest)

        return rules_per_level, av_conf_per_level
