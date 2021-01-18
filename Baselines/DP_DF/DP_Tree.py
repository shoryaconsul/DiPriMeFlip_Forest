''' A class to implement a simple differentially private decision tree (based on ID3) '''

from collections import Counter, defaultdict
import copy
#from pprint import pprint
from . import simple_ml, node

class DP_Tree:
    #_tree = {} # this instance variable becomes accessible to class methods via self._tree
    _attribute_names = []
    _id_tracker = -1
    _root_node = None
    _printed_tree = ""
    _rule_list = []
    _node_list = []
    _num_prunings = 0
    _previous_roots = []

    def __init__(self, instances=None, target_attribute_index=0, trace=0, 
                 max_height=5, min_support=0, epsilon_budget=None,
                 attribute_names=None, epsilon_per_query=0.1, previous_roots=[]):
        self._id_tracker = -1
        self._attribute_names = attribute_names
        self._printed_tree = ""
        self._rule_list = []
        self._node_list = []
        self._num_prunings = 0
        self._previous_roots = previous_roots[:]
        #print( "_previous_roots = "+str(previous_roots) )

        if epsilon_per_query is None:
            if epsilon_budget is not None:
                epsilon_per_query = epsilon_budget / (2*max_height - 1.) # 2 queries per node, except the last level
        #print( "epsilon_per_query = "+str(epsilon_per_query) )

        if instances:
            #self._tree = 
            self._root_node = self._create(instances, [i for i in range(len(instances[0])) if i != target_attribute_index], 
                                           target_attribute_index, trace=trace, max_height=max_height, min_support=min_support, epsilon=epsilon_per_query,
                                      parent_id=-1, parent_value=None, parent_node=None)

    # This is the main recursive function.
    def _create(self, instances, candidate_attribute_indexes, target_attribute_index=0, default_class=None, trace=0, 
                max_height=None, min_support=0, epsilon=1.0, parent_id=-1, parent_value=None, parent_node=None):
        '''
        Returns a new decision tree by recursively selecting and splitting instances based on 
        the highest information_gain of the candidate_attribute_indexes.
        The class label is found in target_attribute_index.
        The default class is the majority value for that branch of the tree.
        A positive trace value will generate trace information with increasing levels of indentation.

        max_height is the maximum levels the tree can have. Assume trace is non-zero.
        min_support is the minimum number of records needed to make a split. Otherwise the node becomes a leaf.
        epsilon budget
    
        Derived from the simplified ID3 algorithm presented in Building Decision Trees in Python by Christopher Roach,
        http://www.onlamp.com/pub/a/python/2006/02/09/ai_decision_trees.html?page=3
        '''
        instances = instances[:]
        self._id_tracker += 1

        class_labels_and_counts = dict(Counter([instance[target_attribute_index] for instance in instances]))
        count_node = sum(class_labels_and_counts.values())
        #print( "class_labels_and_counts = "+str(class_labels_and_counts) )

        if epsilon is not None:
            class_labels_and_counts = simple_ml.add_laplace_noise(class_labels_and_counts, 1., epsilon) # sensitivity = 1
        #print( "new class_labels_counts = "+str(class_labels_and_counts) )

        partitionSize = 0
        for k,v in class_labels_and_counts.items():
            partitionSize += v
        #print("noisy partitionSize = "+str(partitionSize))

        class_label = Counter(class_labels_and_counts).most_common(1)[0][0]

        # If the dataset is empty or the candidate attributes list is empty, return the default value. 
        if partitionSize==0:
            #if trace:
            #    print( '{}Using default class {}'.format('< ' * trace, default_class) )
            return node.Node(self._id_tracker, trace, None, "_Leaf", parent_id, parent_value, parent_node, {default_class:1}, 1, children=None) #default_class
        # If the dataset is empty or the candidate attributes list is empty, return the default value. 
        elif not candidate_attribute_indexes:
            #if trace:
            #    print( '{}Using default class {}'.format('< ' * trace, default_class) )
            return node.Node( self._id_tracker, trace, None, "_Leaf", parent_id, parent_value, parent_node, class_labels_and_counts, count_node, children=None ) #default_class

        # If all the records have the same class label, return that class label
        elif len(Counter(class_labels_and_counts)) == 1:
            #if trace:
            #    print( '{}All {} records have label {}'.format('< ' * trace, partitionSize, class_label) )
            return node.Node(self._id_tracker, trace, None, "_Leaf", parent_id, parent_value, parent_node, class_labels_and_counts, count_node, children=None) #class_label

        # If there aren't enough records in the node to make another split, return the majority class label
        elif partitionSize < min_support:
            #if trace:
            #    print( '{} {} records is below the minimum support required for more splits. The majority label is {}'.format('< ' * trace, partitionSize, class_label) )
            return node.Node(self._id_tracker, trace, None, "_Leaf", parent_id, parent_value, parent_node, class_labels_and_counts, count_node, children=None) #class_label

        # If the tree has reached the maximum number of levels (depth), return the majority class label. Assumes trace is non-zero.
        elif trace >= max_height:
            #if trace:
            #    print( '{}The maximum tree depth has been reached. The {} records in this leaf have majority label {}'.format('< ' * trace, partitionSize, class_label) )
            return node.Node(self._id_tracker, trace, None, "_Leaf", parent_id, parent_value, parent_node, class_labels_and_counts, count_node, children=None)#class_label

        #  MAKE MORE SPLITS
        default_class = class_label #simple_ml.majority_value(instances, target_attribute_index)

        # Choose the next best attribute index to best classify the records
        worst_case_sens = 1. - ( (partitionSize/(partitionSize+1))**2 + (1/(partitionSize+1))**2 )
        #print( "worst_case_sens="+str(worst_case_sens) )
        if trace==1: # if root node
            candi = [i for i in candidate_attribute_indexes if i not in self._previous_roots]
            best_index = simple_ml.choose_best_attribute_index(instances, candi, target_attribute_index, epsilon=epsilon, sensitivity=worst_case_sens)
        else:
            best_index = simple_ml.choose_best_attribute_index(instances, candidate_attribute_indexes, target_attribute_index, epsilon=epsilon, sensitivity=worst_case_sens)
        #if trace:
        #    print( '{}Creating tree node for attribute index {}'.format('> ' * trace, best_index) )

        # Create a new decision tree node with the best attribute index and an empty dictionary object (for now)
        #tree = {best_index:{}}
        current_node = node.Node(self._id_tracker, trace, best_index, self._attribute_names[best_index], parent_id, parent_value, parent_node, class_labels_and_counts, count_node, children=[])

        # Create a new decision tree sub-node (branch) for each of the values in the best attribute field       
        partitions = simple_ml.split_instances(instances, best_index)

        # Remove that attribute from the set of candidates for further splits
        remaining_candidate_attribute_indexes = [i for i in candidate_attribute_indexes if i != best_index]

        ''' For every value in the chosen attribute, make a subtree '''
        tracecopy = trace+1
        curr_id = self._id_tracker
        for attribute_value in partitions:
            #if trace:
            #    print( '{}Creating subtree for value {} ({}, {}, {}, {})'.format(  
            #                        '> ' * trace,
            #                        attribute_value, 
            #                        len(partitions[attribute_value]), 
            #                        len(remaining_candidate_attribute_indexes), 
            #                        target_attribute_index, 
            #                        default_class)
            #                            )

            # Create a subtree for each value of the the best attribute
            subtree = self._create( partitions[attribute_value],
                                    remaining_candidate_attribute_indexes,
                                    target_attribute_index,
                                    default_class,
                                    tracecopy if trace else 0, 
                                    max_height, min_support, epsilon,
                                    curr_id, attribute_value, current_node)

            # Add the new subtree to the empty dictionary object in the new tree/node we just created
            #tree[best_index][attribute_value] = subtree
            current_node.add_child(subtree)
            self._node_list.append(subtree)
            #print('.', end='')
     
        return current_node #tree

        
    # a method intended to be "protected" that can implement the recursive algorithm to classify an instance given a tree
    def _classify(self, node, instance, default_class=None):
        if not node:
            return default_class

        if not node._children: # if leaf
            #print("rec[class]={} & prediction={}".format(instance[0], node._default_prediction))
            return Counter(node._class_counts).most_common(1)[0][0]
        else:
            #print(',', end='')
            attr = node._attribute_index
            rec_val = instance[attr]
            child = None
            for i in node._children:
                if i._parent_value == rec_val:
                    child = i
                    #print("attr={}, rec_val={}, child={}".format(attr, rec_val, child._class_counts))
                    break
            if child is None: # if the record's value couldn't be found:
                return Counter(node._class_counts).most_common(1)[0][0]
            return self._classify(child, instance, default_class)


    def classify_list(self, instances, default_class=None):
        return [self._classify(self._root_node, instance, default_class) for instance in instances]


    def evaluate_accuracy(self, instances, default_class=None, target_index=0):
        ''' Calculate the Prediction Accuracy of the class. '''
        actual_labels = [x[target_index] for x in instances]
        predicted_labels = self.classify_list(instances, default_class)
        counts = Counter([x == y for x, y in zip(predicted_labels, actual_labels)])
        return float(counts[True]) / len(instances) #counts[True], counts[False], float(counts[True]) / len(instances)
         
    # recursive function for outputting a description of each node.
    def _print_node(self, node):
        node_string = "{}->{}\n{} Level {}, ID {}: entering value -> {} & entering class counts -> {} (gini={:.3}) # splitting attribute {}: {}".format(
            node._parent_id, node._id, '~~~'*node._level, node._level, node._id, node._parent_value, node._class_counts, 
            node._gini if node._gini else 1., node._attribute_index, node._attribute_name)
        #print(node_string)
        self._printed_tree += node_string+"\n"

        if node._children:
            for child in node._children:
                self._print_node(child)


    def print_tree(self):
        self._print_node(self._root_node)
        return self._printed_tree


    def prune_tree(self):
        self._traverse_down(self._root_node, pruning=True)
        return self._num_prunings
    

    def _prune_leaf(self, parent_of_leaf_node):
        if parent_of_leaf_node._children:
            average_gini = 0.
            for child in parent_of_leaf_node._children:
                average_gini += child._gini * child._node_size/float(parent_of_leaf_node._node_size)
            #print("average_gini vs. parent_gini = {:.4} vs. {:.4}".format(average_gini, parent_of_leaf_node._gini))
            # # # 10% THRESHOLD INCLUDED # # #
            if average_gini*1.0 >= parent_of_leaf_node._gini: 
                parent_of_leaf_node._children = None
                parent_of_leaf_node._attribute_index = None
                parent_of_leaf_node._attribute_name = "_Pruned_Leaf"
                self._num_prunings += 1
                if parent_of_leaf_node._parent_node._parent_node:
                    self._prune_leaf(parent_of_leaf_node._parent_node) 


    def _traverse_down(self, node, pruning=False):
        if node._children:
            for child in node._children:
                self._traverse_down(child, pruning=pruning)  
        elif not pruning:
            self._build_rule(node)
        
        if node._children and pruning and node._parent_node:
            if node._parent_node._parent_node:
                no_grandchildren = [not child._children for child in node._children]
                #print("children of children of node"+str(node._id)+": "+str(answers))
                # we start at the second-lowest level, and Level 2 cannot be pruned to Level 1:
                if all(no_grandchildren):
                    self._prune_leaf(node)

           
    def _build_rule(self, leaf_node):
        ''' Recursively climbs the tree from a leaf and adds a dictionary for each node in the rule. 
        The dictionaries combine to form a single rule. '''
        curr_node = leaf_node
        built_rule = []
        while curr_node._parent_node:
            node_summary = {"attribute_index":curr_node._parent_node._attribute_index, 
                            "attribute_name":curr_node._parent_node._attribute_name, 
                            "parent_value":curr_node._parent_value, 
                            "class_counts":curr_node._class_counts, 
                            "gini":curr_node._gini, 
                            "prediction":Counter(curr_node._class_counts).most_common(1)[0][0]}
            built_rule.append(node_summary)
            curr_node = curr_node._parent_node
        self._rule_list.append(built_rule)
                

    def _filter_rule_list(self):
        unique_rules = []
        for rule in self._rule_list: # each rule is a list of dictionaries
            #print("current rule = "+str(rule))
            nodes = []
            for node in rule:
                nodes.append([node["attribute_index"], node["parent_value"]])
            
            new_rule = True
            for u_rule in unique_rules:
                nodes1 = []
                for node1 in u_rule:
                    nodes1.append([node1["attribute_index"], node1["parent_value"]])
                if all([n in nodes1 for n in nodes]):
                    new_rule = False
                    break
            if new_rule:        
                unique_rules.append(rule)

        return unique_rules


    def _convert_rule_to_text(self, rule):
        text = ""
        for node in rule[::-1]:
            text += "{}={} {}(gini={:.3}) &&& ".format(
                node["attribute_name"], node["parent_value"], node["class_counts"], node["gini"])
        text = text[:-5] 
        text += " --> {}".format(rule[0]["prediction"])
        return text


    def print_rule_list(self):
        self._traverse_down(self._root_node)
        #self._build_rule( self._root_node, [{"attribute_index":None, "attribute_name":None, "parent_value":None, "gini":None, "prediction":None},] )
        #print("unfiltered rule list size = "+str(len(self._rule_list)))
        filtered = self._filter_rule_list()
        #print("new rule list size = "+str(len(filtered)))
        rule_list_text = []
        for rule in filtered:
            rule_text = self._convert_rule_to_text(rule)
            #print(rule_text)
            rule_list_text.append(rule_text)
        return rule_list_text

    
    def _collect_rules(self, node):
        all_rules = [node]
        if node._children:
            for child in node._children:
                all_rules.extend(self._collect_rules(child))
        return all_rules

    def rules_and_conf_per_level(self):
        all_subsets_of_rules = self._collect_rules(self._root_node)
        rules_per_level = defaultdict(int)
        av_conf_per_level = defaultdict(float)
        for node in all_subsets_of_rules:
            rules_per_level[node._level] += 1
            av_conf_per_level[node._level] += node._confidence
        for k,v in av_conf_per_level.items():
             av_conf_per_level[k] = v/rules_per_level[k]
        return rules_per_level, av_conf_per_level