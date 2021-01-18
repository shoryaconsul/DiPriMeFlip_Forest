''' Utility functions to implement some simple Machine Learning tasks '''
import math
import numpy as np
import bisect
import operator
from scipy import stats
from collections import defaultdict, Counter
import random

def load_instances(filename, filter_missing_values=False, missing_value='?'):
    '''Returns a list of instances (records) stored in a file.
    
    filename is expected to have a series of comma-separated attribute values per line, e.g.,
        p,k,f,n,f,n,f,c,n,w,e,?,k,y,w,n,p,w,o,e,w,v,d'''
    instances = []
    with open(filename, 'r') as f:
        for line in f:
            new_instance = line.strip().split(', \t;')
            if not filter_missing_values or missing_value not in new_instance:
                instances.append(new_instance)
    return instances


def save_instances(filename, instances):
    '''Saves a list of instances (records) to a file.
    
    instances are saved to filename one per line, 
    where each instance is a list of attribute value strings.'''
    with open(filename, 'w') as f:
        for instance in instances:
            f.write(','.join(instance) + '\n')


def load_attribute_names(filename, separator=':'):
    '''Returns a list of attribute names in a file.
    
    filename is expected to be a file with attribute names. one attribute per line.
    
    filename might also have a list of possible attribute values, in which case it is assumed
    that the attribute name is separated from the possible values by separator.'''
    with open(filename, 'r') as f:
        attribute_names = [line.strip().split(separator)[0] for line in f]
    return attribute_names


def load_attribute_values(attribute_filename):
    '''Returns a list of attribute values in filename.
    
    The attribute values are represented as dictionaries, 
    wherein the keys are abbreviations and the values are descriptions.
    
    filename is expected to have one attribute name and set of values per line, 
    with the following format:
        name: value_description=value_abbreviation[,value_description=value_abbreviation]*
    for example
        cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    The attribute value description dictionary created from this line would be the following:
        {'c': 'conical', 'b': 'bell', 'f': 'flat', 'k': 'knobbed', 's': 'sunken', 'x': 'convex'}'''
    attribute_values = []
    with open(attribute_filename) as f:
        for line in f:
            attribute_name_and_value_string_list = line.strip().split(':')
            attribute_name = attribute_name_and_value_string_list[0]
            if len(attribute_name_and_value_string_list) < 2:
                attribute_values.append({}) # no values for this attribute
            else:
                value_abbreviation_description_dict = {}
                description_and_abbreviation_string_list = attribute_name_and_value_string_list[1].strip().split(',')
                for description_and_abbreviation_string in description_and_abbreviation_string_list:
                    description_and_abbreviation = description_and_abbreviation_string.strip().split('=')
                    description = description_and_abbreviation[0]
                    if len(description_and_abbreviation) < 2: # assumption: no more than 1 value is missing an abbreviation
                        value_abbreviation_description_dict[None] = description
                    else:
                        abbreviation = description_and_abbreviation[1]
                        value_abbreviation_description_dict[abbreviation] = description
                attribute_values.append(value_abbreviation_description_dict)
    return attribute_values


def load_attribute_names_and_values(filename):
    '''Returns a list of attribute names and values in filename.
    
    This list contains dictionaries wherein the keys are names 
    and the values are value description dictionaries.
    
    Each value description sub-dictionary will use the attribute value abbreviations as its keys 
    and the attribute descriptions as the values.
    
    filename is expected to have one attribute name and set of values per line, with the following format:
        name: value_description=value_abbreviation[,value_description=value_abbreviation]*
    for example
        cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    The attribute name and values dictionary created from this line would be the following:
        {'name': 'cap-shape', 'values': {'c': 'conical', 'b': 'bell', 'f': 'flat', 'k': 'knobbed', 's': 'sunken', 'x': 'convex'}}'''
    attribute_names_and_values = [] # this will be a list of dicts
    with open(filename) as f:
        for line in f:
            attribute_name_and_value_dict = {}
            attribute_name_and_value_string_list = line.strip().split(':')
            attribute_name = attribute_name_and_value_string_list[0]
            attribute_name_and_value_dict['name'] = attribute_name
            if len(attribute_name_and_value_string_list) < 2:
                attribute_name_and_value_dict['values'] = None # no values for this attribute
            else:
                value_abbreviation_description_dict = {}
                description_and_abbreviation_string_list = attribute_name_and_value_string_list[1].strip().split(',')
                for description_and_abbreviation_string in description_and_abbreviation_string_list:
                    description_and_abbreviation = description_and_abbreviation_string.strip().split('=')
                    description = description_and_abbreviation[0]
                    if len(description_and_abbreviation) < 2: # assumption: no more than 1 value is missing an abbreviation
                        value_abbreviation_description_dict[None] = description
                    else:
                        abbreviation = description_and_abbreviation[1]
                        value_abbreviation_description_dict[abbreviation] = description
                attribute_name_and_value_dict['values'] = value_abbreviation_description_dict
            attribute_names_and_values.append(attribute_name_and_value_dict)
    return attribute_names_and_values
    
    
def attribute_values(instances, attribute_index):
    '''Returns the distinct values of an attribute across a list of instances.
    
    instances is expected to be a list of instances (attribute values).
    attribute_index is expected bo be a the position of attribute in instances.
    
    See http://www.peterbe.com/plog/uniqifiers-benchmark for variants on this algorirthm'''
    return list(set([x[attribute_index] for x in instances]))


def attribute_value(instance, attribute, attribute_names):
    '''Returns the value of an attribute in an instance.
    
    Based on the position of attribute in the list of attribute_names'''
    if attribute not in attribute_names:
        return None
    else:
        i = attribute_names.index(attribute)
        return instance[i] # using the parameter name here
        

def print_attribute_names_and_values(instance, attribute_names):
    '''Prints the attribute names and values for instance'''
    print( 'Values for the', len(attribute_names), 'attributes:\n' )
    for i in range(len(attribute_names)):
        print( attribute_names[i], '=', attribute_value(instance, attribute_names[i], attribute_names) )


def attribute_value_counts(instances, attribute, attribute_names):
    '''Returns a defaultdict containing the counts of occurrences of each value of attribute in the list of instances.
    attribute_names is a list of names of attributes.'''
    i = attribute_names.index(attribute)
    value_counts = defaultdict(int)
    for instance in instances:
        value_counts[instance[i]] += 1
    return value_counts


def print_all_attribute_value_counts(instances, attribute_names):
    '''Returns a list of defaultdicts containing the counts of occurrences of each value of each attribute in the list of instances.
    attribute_names is a list of names of attributes.'''
    num_instances = len(instances) * 1.0
    for attribute in attribute_names:
        value_counts = attribute_value_counts(instances, attribute, attribute_names)
        print( '{}:'.format(attribute), )
        for value, count in sorted(value_counts.iteritems(), key=operator.itemgetter(1), reverse=True):
            print( '{} = {} ({:5.3f}),'.format(value, count, count/num_instances), )
        print  
        
    
def entropy(instances, class_index=0, attribute_name=None, value_name=None):
    '''Calculate the entropy of attribute in position attribute_index for the list of instances.'''
    num_instances = len(instances)
    if num_instances <= 1:
        return 0
    value_counts = defaultdict(int)
    for instance in instances:
        value_counts[instance[class_index]] += 1
    num_values = len(value_counts)
    if num_values <= 1:
        return 0
    attribute_entropy = 0.0
    n = float(num_instances)
    if attribute_name:
        print( 'entropy({}{}) = '.format(attribute_name, '={}'.format(value_name) if value_name else '') )
    for value in value_counts:
        value_probability = value_counts[value] / n
        child_entropy = value_probability * math.log(value_probability, num_values)
        attribute_entropy -= child_entropy
        if attribute_name:
            print( '  - p({0}) x log(p({0}), {1})  =  - {2:5.3f} x log({2:5.3f})  =  {3:5.3f}'.format(
                value, num_values, value_probability, child_entropy) )
    if attribute_name:
        print( '  = {:5.3f}'.format(attribute_entropy) )
    return attribute_entropy


def information_gain(instances, parent_index, class_index=0, attribute_name=False):
    '''Return the information gain of splitting the instances based on the attribute parent_index'''
    parent_entropy = entropy(instances, class_index, attribute_name)
    child_instances = defaultdict(list)
    for instance in instances:
        child_instances[instance[parent_index]].append(instance)
    children_entropy = 0.0
    n = float(len(instances))
    for child_value in child_instances:
        child_probability = len(child_instances[child_value]) / n
        children_entropy += child_probability * entropy(child_instances[child_value], class_index, attribute_name, child_value)
    return parent_entropy - children_entropy
    

def gini(records, candidate_index, class_index=0):
    '''Return the gini index of the instances filtered with the candidate_index compared to the gini index of all the instances'''
    class_counts = defaultdict(int) # class values and their count
    value_counts = defaultdict(int) # candidate attribute values and their count
    for rec in records:
        class_counts[rec[class_index]] += 1
        value_counts[rec[candidate_index]] += 1
   
    current = 0.
    for c in class_counts:
        class_probability = class_counts[c] / len(records)
        current += class_probability ** 2
    node_gini = 1. - current  

    candidate_current = 0.
    for val in value_counts:
        candidate_class_counts = defaultdict(int) # class values for each attribute value and their count
        for rec in records:
            if rec[candidate_index]==val:
                candidate_class_counts[rec[class_index]] += 1

        value_current = 0.
        for c in candidate_class_counts:
            value_class_probability = candidate_class_counts[c] / value_counts[val]
            value_current += value_class_probability ** 2
        candidate_current += value_counts[val]/len(records) * (1. - value_current)

    candidate_gini = -(candidate_current - node_gini)
    #print("node_gini = {:.4}, candidate = {:.4} --> gini improvement = {:.4}".format(node_gini, candidate_current, candidate_gini))
    return candidate_gini

def max_criteria(records, candidate_index, class_index=0):
    '''Return the "max" splitting criteria of the instances filtered with the candidate_index'''
    class_counts = defaultdict(int) # class values and their count
    value_counts = defaultdict(int) # candidate attribute values and their count
    for rec in records:
        class_counts[rec[class_index]] += 1
        value_counts[rec[candidate_index]] += 1


def majority_value(instances, class_index=0):
    '''Return the most frequent value of class_index in instances'''
    class_counts = Counter([instance[class_index] for instance in instances])
    return class_counts.most_common(1)[0][0]


def choose_best_attribute_index(instances, candidate_attribute_indexes, class_index=0, epsilon=1.0, sensitivity=1.0, numers=[]):
    '''Return the index of the attribute that will provide the greatest information gain 
        if instances were partitioned based on that attribute'''
    if epsilon is None:
        gains_and_indexes = sorted([ (gini(instances, i, class_index), i) for i in candidate_attribute_indexes], reverse=True) # or information_gain
        return gains_and_indexes[0][1] # each element has 2 components; the second one is the index of the attribute

    ginis = [ [gini(instances, i, class_index), i] for i in candidate_attribute_indexes if i not in numers]
    splits = {}
    #print("TESTING NUMERS")
    for i in numers:
        if i in candidate_attribute_indexes:
            info = best_numer_gini(instances, i, class_index=class_index, epsilon=epsilon) # = [best_gini, candidate_index, best_split]
            splits[str(i)] = info[2]
            ginis.append([info[0], i])
    #print("ginis = "+str(ginis))
    weighted = []
    for g,i in ginis:
        power = min( 50, (epsilon*g)/(2*sensitivity) )
        weighted.append( [math.exp(power), i] )
    #print("weighted = "+str(weighted))
    sum = 0.
    for g,i in weighted:
        sum += g
    for wi in range(len(weighted)):
        weighted[wi][0] /= sum   
    #print("super weighted = "+str(weighted))
    customDist = stats.rv_discrete(name='customDist', values=([i for g,i in weighted], [g for g,i in weighted]))
    best_att = customDist.rvs()
    split = None
    if best_att in numers:
        split = splits[str(best_att)]
    #print("best_att examples = "+str(customDist.rvs(size=20)))
    return best_att#, split


def choose_best_binary_split(instances, candidate_attribute_indexes, numer_indexes, class_index=0):
    ''' Return the best attribute and splitting value for making a binary split in the partition. '''
    ginis_and_values = []
    ginis_and_values.extend([best_categ_gini(instances, i, class_index) for i in candidate_attribute_indexes if i not in numer_indexes])
    ginis_and_values.extend([best_numer_gini(instances, i, class_index) for i in candidate_attribute_indexes if i in numer_indexes])
    sorted_ginis = sorted(ginis_and_values, key=lambda x: x[0], reverse=True)
    #print("sorted_ginis = "+str(sorted_ginis))
    return sorted_ginis[0][1], sorted_ginis[0][2]


def best_numer_gini(records, candidate_index, class_index=0, epsilon=1.0):
    ''' Return the best gini index and corresponding split point of the numerical attribute. '''
    size_D = float(len(records))
    values = sorted([float(rec[candidate_index]) for rec in records])
    min_domain = values[0]
    max_domain = values[-1]
    splits = []
    for i in range(1, len(values), max(1,math.ceil(len(values)/30.))): # skip some splits to speed up the program
        if values[i] > values[i-1]:
            splits.append( (values[i-1]+values[i])/2. )

    #best_gini = 0.
    #best_split = 0.
    #print("values:{} splits:{}".format(len(values), len(splits)))
    if len(splits)<2:
        return [0.0, candidate_index, (min_domain+max_domain)/2.]
    else:
        gini_info = []
        range_min = min_domain
        prev_gini = -99.
        for i,split in enumerate(splits):
            index = bisect.bisect_right(values, split)
            above_count = len(values[index:len(values)])
            below_count = size_D - above_count
            #print("split="+str(split)+"  above_count="+str(above_count))

            above_class_counts = defaultdict(int) # class values for the value and their count
            below_class_counts = defaultdict(int) # class values for every other value and their count
            for rec in records:
                if float(rec[candidate_index]) > split:
                    above_class_counts[rec[class_index]] += 1
                else:
                    below_class_counts[rec[class_index]] += 1
         
            sum_of_above_classes = sum( [(above_class_counts[c]/above_count)**2 for c in above_class_counts])
            sum_of_below_classes = sum( [(below_class_counts[c]/below_count)**2 for c in below_class_counts])

            weighted_above_gini = above_count/size_D * sum_of_above_classes # \frac{|D_>s|}{|D|} * sum
            weighted_below_gini = below_count/size_D * sum_of_below_classes # \frac{|D_<=s|}{|D|} * sum
            total_split_gini = weighted_above_gini + weighted_below_gini

            #print("{}: curr:{} prev:{}".format(i, total_split_gini, prev_gini))
            if i>0 and total_split_gini != prev_gini: # then we know the previous gini range is "finished"
                if i!=len(splits)-1: range_max = split
                else: range_max = max_domain
                range_full = range_max - range_min
                gini_info.append({"gini":prev_gini, "split": random.uniform(range_min,range_max), "range":range_full}) # add the previous gini
                range_min = range_max # the minimum for the next split
                prev_gini = total_split_gini # for the next gini range
            elif i==0: # if we haven't added any gini's yet, just update the gini
                prev_gini = total_split_gini
            elif i==len(splits)-1:
                gini_info.append({"gini":total_split_gini, "split": random.uniform(range_min,max_domain), "range":max_domain-range_min}) # add the final gini

            #if total_split_gini > best_gini: # higher is better
            #    best_gini = total_split_gini
            #    best_split = split

        best_split, best_gini = expo_mech(epsilon, 0.5, gini_info)
        #best_gini = -99.
        #for info in gini_info:
        #    print("best_split:{} curr_split:{}".format(best_split, info["split"]))
        #    if info["split"]==best_split: 
        #        best_gini = info["gini"]
        #        break
        #if best_gini==-99.: print("ERROR: gini still -99 for split {}".format(best_split))
        #print("BEST SPLIT FOR {} = {} with gini = {}".format(candidate_index, best_split, best_gini))
        return [best_gini, candidate_index, best_split]


def expo_mech(e, s, gini_info):
    weighted = [] 
    #reference = {}
    for i,triplet in enumerate(gini_info):
        score = triplet["gini"]
        #split_val = str(triplet["split"])
        domain = triplet["range"]
        #reference[str(split_val)] = score
        power = min( 50, (e*score)/(2*s) )
        weighted.append( [i, math.exp(power)*domain] ) 
        if domain<0.0001: print("exp({})*{} ~ Weighted: {}".format(power, domain, weighted[-1]), end='\t')
    sum = 0.
    for pair in weighted:
        sum += pair[1]
    for i in range(len(weighted)):
        weighted[i][1] /= sum   
    #print(weighted)
    customDist = stats.rv_discrete(name='customDist', values=([pair[0] for pair in weighted], [pair[1] for pair in weighted]))
    split_index = customDist.rvs()
    #print("best_att examples = "+str(customDist.rvs(size=20)))
    return float(gini_info[split_index]["split"]), gini_info[split_index]["gini"]


def best_categ_gini(records, candidate_index, class_index=0):
    '''Return the best gini index and corresponding value of the categorical attribute. '''
    size_D = float(len(records))
    #class_counts = defaultdict(int) # class values and their count
    value_counts = defaultdict(int) # candidate attribute values and their count
    all_value_class_counts = {}
    for rec in records:
        #class_counts[rec[class_index]] += 1
        value_counts[rec[candidate_index]] += 1
        if rec[candidate_index] not in all_value_class_counts:
            all_value_class_counts[rec[candidate_index]] = defaultdict(int)
        all_value_class_counts[rec[candidate_index]][rec[class_index]] += 1

    #current = 0.
    #for c in class_counts:
    #    class_probability = class_counts[c] / len(records)
    #    current += class_probability ** 2
    #node_gini = 1. - current # the starting gini

    best_gini = 0.
    best_value = None
    for val in value_counts:
        value_class_counts = all_value_class_counts[val] # class values for the value and their count
        not_value_class_counts = defaultdict(int) # class values for every other value and their count
        size_D_v = float(value_counts[val])
        for value,counts in all_value_class_counts.items():
            if value != val:
                for cls,count in counts.items():
                    not_value_class_counts[cls] += 1

        sum_of_value_classes = sum( [(value_class_counts[c]/size_D_v)**2 for c in value_class_counts]) # \frac{D_v,c}{D_v}
        sum_of_not_value_classes = sum( [(not_value_class_counts[c]/(size_D-size_D_v))**2 for c in not_value_class_counts]) # \frac{D_!v,c}{D_!v}
        
        weighted_value_gini = size_D_v/size_D * sum_of_value_classes # \frac{D_v}{D} * sum
        weighted_not_value_gini = (size_D-size_D_v)/size_D * sum_of_not_value_classes # \frac{D_!v}{D} * sum
        total_value_gini = weighted_value_gini + weighted_not_value_gini
        if total_value_gini > best_gini: # higher is better
            best_gini = total_value_gini
            best_value = val
    #candidate_gini = -(candidate_current - node_gini)
    #print("node_gini = {:.4}, candidate = {:.4} --> gini improvement = {:.4}".format(node_gini, candidate_current, candidate_gini))
    return [best_gini, candidate_index, best_value] #candidate_gini


def add_laplace_noise(counts, sensitivity, epsilon):
    for k, v in counts.items():
        counts[k] = max(0, int(v + np.random.laplace( scale=float(sensitivity/epsilon) )))
    return counts


def cmp_partitions(p1, p2):
    if entropy(p1) < entropy(p2):
        return -1
    elif entropy(p1) > entropy(p2):
        return 1
    elif len(p1) < len(p2):
        return -1
    elif len(p1) > len(p2):
        return 1
    return 0


def split_instances(instances, attribute_index, split=None):
    '''Returns a list of dictionaries, splitting a list of instances according to their values of a specified attribute''
    
    The key of each dictionary is a distinct value of attribute_index,
    and the value of each dictionary is a list representing the subset of instances that have that value for the attribute'''
    partitions = defaultdict(list)
    if not split: # not continuous attribute
        for instance in instances:
            partitions[instance[attribute_index]].append(instance)
    else:
        for instance in instances:
            if instance[attribute_index] < split:
                partitions["<"].append(instance)
            else: # >=
                partitions[">="].append(instance)
    return partitions


def split_instances_binary(records, attribute_index, split_value, numer=False):
    ''' Split the records into 2 partitions. Handles both numerical and categorical splits. '''
    above = [] # or "is value"
    below = [] # or "is not value"
    if numer:
        for rec in records:
            if float(rec[attribute_index]) > split_value:
                above.append(rec)
            else:
                below.append(rec)
    else: # categorical
        for rec in records:
            if rec[attribute_index] == split_value:
                above.append(rec)
            else:
                below.append(rec)
    return [above,below]


def partition_instances(instances, num_partitions):
    '''Returns a list of relatively equally sized disjoint sublists (partitions) of the list of instances'''
    return [[instances[j] for j in xrange(i, len(instances), num_partitions)] for i in xrange(num_partitions)]


def classify(tree, instance, default_class=None):
    '''Returns a classification label for instance, given a decision tree'''
    if not tree:
        return default_class
    if not isinstance(tree, dict): 
        return tree
    attribute_index = tree.keys()[0]
    attribute_values = tree.values()[0]
    instance_attribute_value = instance[attribute_index]
    if instance_attribute_value not in attribute_values:
        return default_class
    return classify(attribute_values[instance_attribute_value], instance, default_class)


def classification_accuracy(tree, testing_instances, class_index=0):
    '''Returns the accuracy of classifying testing_instances with tree, 
    where the class label is in position class_index'''
    num_correct = 0
    for i in xrange(len(testing_instances)):
        prediction = classify(tree, testing_instances[i])
        actual_value = testing_instances[i][class_index]
        if prediction == actual_value:
            num_correct += 1
    return float(num_correct) / len(testing_instances)
    

def compute_learning_curve(instances, num_partitions=10):
    '''Returns a list of training sizes and scores for incrementally increasing partitions.
    
    The list contains 2-element tuples, each representing a training size and score.
    The i-th training size is the number of instances in partitions 0 through num_partitions - 2.
    The i-th score is the accuracy of a tree trained with instances 
    from partitions 0 through num_partitions - 2
    and tested on instances from num_partitions - 1 (the last partition).'''
    partitions = partition_instances(instances, num_partitions)
    testing_instances = partitions[-1][:]
    training_instances = partitions[0][:]
    accuracy_list = []
    for i in xrange(1, num_partitions):
        # for each iteration, the training set is composed of partitions 0 through i - 1
        tree = create_decision_tree(training_instances)
        partition_accuracy = classification_accuracy(tree, testing_instances)
        accuracy_list.append((len(training_instances), partition_accuracy))
        training_instances.extend(partitions[i][:])
    return accuracy_list