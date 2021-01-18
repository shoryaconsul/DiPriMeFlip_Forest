from collections import Counter, defaultdict

ABOVE_SPLIT = 1 # numerical
BELOW_SPLIT = -1 # numerical
IS_VALUE = 2 # categorical
NOT_VALUE = -2 # categorical

class Node:
    """ A data structure for decision trees. """

    #_id = None # the unique id of this node
    #_level = None
    #_attribute_index = None # attribute that this node will split on
    #_binary_split_value = None # value that this node will split on, if binary tree
    #_binary_split_type = None # attribute split type of this node, if binary tree
    #_attribute_name = "None" # the name of the attribute that this node will split on
    #_parent_id = None # id of this node's parent
    #_parent_value = None # attribute value that led to this node
    #_parent_split_type = None # attribute split type that led to this node
    #_parent_node = None # the parent node
    #_class_counts = {} # dictionary of value:count pairs
    #_children = [] # list of nodes
    #_gini = None # lower is better
    #_node_size = None # support
    #_confidence = None

    #_new_node_size = [] # support when filtering different data through the tree, in order of different noise levels
    #_new_class_counts = []


    def __init__(self, id, level, attribute_index, attribute_name, parent_id, parent_value, parent_node, class_counts, count_node, 
                 children=[], parent_split_type=IS_VALUE, binary_split_type=IS_VALUE, binary_split_value=None, svfp_numer=None):
        self._id = id
        self._level = level
        self._attribute_index = attribute_index
        self._attribute_name = attribute_name
        self._parent_id = parent_id
        if not parent_value: # root
            self._parent_value = None
        elif abs(parent_split_type)>1: # categorical
            self._parent_value = str.strip(str(parent_value))
        else:
            self._parent_value = float(parent_value)
        self._parent_split_type = parent_split_type
        self._parent_node = parent_node
        self._class_counts = class_counts
        self._count = count_node
        self._children = children

        self._svfp_numer = svfp_numer # for numericals in Friedman

        self._binary_split_type = binary_split_type
        self._binary_split_value = binary_split_value

        self._new_node_size = [] # support when filtering different data through the tree 
        self._new_class_counts = []

        if class_counts and sum([v for k,v in class_counts.items()])>0:
            self._gini = self._calc_gini(class_counts)
            majority = Counter(class_counts).most_common(1)[0][1]
            self._noisy_majority = Counter(class_counts).most_common(1)[0][0] # the value itself
            partitionSize = 0
            for k,v in class_counts.items():
                partitionSize += v
            self._node_size = partitionSize
            self._confidence = majority / float(self._node_size)
        else:
            self._noisy_majority = None
            majority = 0.
            self._gini = 1.
            self._node_size = 0
            self._confidence = 0.     


    def add_child(self, child):
        self._children.append(child)

    def _calc_gini(self, class_counts):
        total = sum([v for k,v in class_counts.items()])
        current = 0.
        for c in class_counts:
            class_probability = class_counts[c] / float(total)
            current += class_probability ** 2
        return 1. - current # lower is better

    #def get_class_counts(self):
    #    if not self._class_counts:
    #        self._class_counts = self._sum_child_classes(defaultdict(int))
    #    print("node class_counts = {}".format(self._class_counts))
    #    return self._class_counts

    #def _sum_child_classes(self, class_counts):
    #    if self._children:
    #        for child in self._children:
    #            class_counts = child._sum_child_classes(class_counts)
    #    else:
    #        if self._class_counts:
    #            for k,v in self._class_counts.items():
    #                class_counts[k] += v
    #    return class_counts