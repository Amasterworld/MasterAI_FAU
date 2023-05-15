# We implement Bayesian networks.
# For simplificity, all probability variables are Booleans.
#
# The networks are represented as Python dictionaries. Below is a sketch
# of the network for the burglary example from the lecture notes.
# It states, for example, that P(alarm | burglary, not earthquake) = 0.94.
# It follows that $P(not alarm | burglary, not earthquake) = 1-0.94.
#
# example_network = {
#     'Burglary': {'name': 'Burglary', 'parents': [], 'probabilities': {(): 0.001}},
#     'Earthquake': {'name': 'Earthquake', 'parents': [], 'probabilities': {(): 0.002}},
#     'Alarm': {
#         'name': 'Alarm',
#         'parents': ['Burglary', 'Earthquake'],
#         'probabilities': {
#             (True, True): 0.95,
#             (True, False): 0.94,
#             (False, True): 0.29,
#             (False, False): 0.001}
#         },
#     'JohnCalls': {'name': 'JohnCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.9, (False,): 0.05}},
#     'MaryCalls': {'name': 'MaryCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.7, (False,): 0.01}}
# }
#
# Queries consist of the network (as above), a single query variable, and an atomic event for the evidence.
# The latter is a dictionary that gives for the every evidence variable a value.
#
# Example query: query(example_network, 'Burglary', {'MaryCalls':True, 'JohnCalls':True})
from itertools import product

def compute_joint_distribution(network):
    """Compute the full joint probability distribution of a Bayesian network.

    Args:
        network (dict): The Bayesian network represented as a dictionary of nodes.

    Returns:
        dict: A dictionary of tuples (corresponding to variable assignments) and their corresponding probabilities.
    """
    # Compute the set of all variables in the network
    all_vars = set()
  
    for node_name, node in network.items():
        all_vars.add(node_name)
        all_vars.update(node["parents"])
    
    # Initialize the joint distribution table
    joint_distribution = {}

    # Generate all possible variable assignments
    for assignment in product([True, False], repeat=len(all_vars)):
        # Convert the assignment tuple to a dictionary
        assignment_dict = {var: val for var, val in zip(all_vars, assignment)}
        #print("I am dic assi ", assignment_dict)
        # Compute the joint probability of this assignment
        joint_probability = 1.0
        for node_name, node in network.items():
            node_parents = node["parents"]
            node_probabilities = node["probabilities"]
            #print ("node_parents", node_parents)
            if node_parents:
                # Compute the conditional probability of the node given its parents
                parent_assignments = tuple(assignment_dict[parent] for parent in node_parents)
                node_probability = node_probabilities[parent_assignments]
                #print ("if it is not root ", node_name, parent_assignments)
                #print ("node_probability", node_probability)
            else:
                # The node has no parents, so just use its unconditional probability
                node_probability = node_probabilities

            # Multiply the joint probability by the probability of this node given its parents
            
            #if it is parents
            if () in node_probabilities:
                #print ("node_probability", node_probability)
                
                joint_probability *= node_probabilities[()]
            else:
                #print ("node_probabilities ", node_probability)
                joint_probability *= node_probability

        # Add the joint probability to the joint distribution table
        joint_distribution[tuple(assignment_dict.items())] = joint_probability
        
        #print(joint_probability)
    return joint_distribution

def query(network, node, evidence):

    
    joint_distribution = compute_joint_distribution(network)
    print(len(joint_distribution))
    #for key, value in joint_distribution.items():
    #    print(f"{key} : {value}")
    #smallest_value = min(joint_distribution.values())
    #for ky, value in joint_distribution.items():
    #    if abs(value - 0.008321327685349975) < 1e-10:
    #        print("value ",value)
    #return abs(smallest_value - 0.008321327685349975) < 1e-10
    
    joint_distribution = compute_joint_distribution(network)

    #for k, v in joint_distribution.items():
    #    for variable, value in k:
    #        if variable == node:
    #            print(variable, v)
                #normalizing_factor = sum(relevant_assignments.values())
    #relevant_assignments = {k: v for k, v in joint_distribution.items() if k[node] == evidence[node]}
        #print(" relevant_assignments ",relevant_assignments)
    #total_probability = sum(relevant_assignments.values())
    
    #filtered_assignments = {k: v/total_probability for k, v in relevant_assignments.items()}
    joint_distribution = compute_joint_distribution(network)

    for k, v in joint_distribution.items():
        print(k,v)
    
    
    smallest = min(joint_distribution.values())
    
    return smallest
    return filtered_assignments






# def query(network, node, evidence):
#   return 0.5    # TODO: compute actual value
# need to compute P(node = true | evidence)
# marginalize over all hidden variables (i.e., all variables in network except node (= query variable) and evidence)
# yields big formula with lots of conditional probabilities
# simplify the formula by throwing out conditions according to the network
# fill in concrete values for the remaining conditional probabilities as given by the network
