import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import time
from HB_function import *
from scipy.stats import norm,gamma
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import random
import tensorflow as tf
import itertools
import scipy.io as sio
from scipy.special import erf


class RipsComplex:     
    def __init__(self, num_points):
        self.parent = list(range(num_points))
        self.rank = [0] * num_points

    def find(self, point):   
        if self.parent[point] != point:
            self.parent[point] = self.find(self.parent[point])
        return self.parent[point]

    def union(self, point1, point2): 
        root1 = self.find(point1)
        root2 = self.find(point2)

        if root1 != root2:  
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def HKCC(edge_pairs, node_pairs, terminals):
    valid_nodes = {node for node, lifetime in node_pairs if lifetime > 0}

    distances = []
    for u, v, t_e in edge_pairs:
        if t_e == float('inf'):  
            distances.append(0)
        elif t_e > 0: 
            distances.append(1 / t_e)
        else:  
            distances.append(float('inf'))
    
    finite_distances = [d for d in distances if d < float('inf')]
    r = max(finite_distances) if finite_distances else 0

    Rips = RipsComplex(len(valid_nodes)) 
    point_to_index = {node: i for i, node in enumerate(valid_nodes)}
  
    for i, (point1, point2, _) in enumerate(edge_pairs):
        d_e = distances[i] 
        
        if (point1 in valid_nodes and 
            point2 in valid_nodes and 
            d_e <= r):
            Rips.union(point_to_index[point1], point_to_index[point2])
    
    root_set = set()
    for terminal in terminals:
        index = point_to_index[terminal]
        root_set.add(Rips.find(index))
    return len(root_set) == 1


def split_nodes(sorted_nodes):
    midpoint = (len(sorted_nodes) + 1) // 2
    S1 = sorted_nodes[:midpoint]
    S2 = sorted_nodes[midpoint:]
    return S1, S2


def K_terminal_pair(new_edge_pairs, new_node_pairs, terminals):
    unified = []
    for u, t in new_node_pairs:
        unified.append(('node', u, t))
    for idx, (u, v, t) in enumerate(new_edge_pairs):
        unified.append(('edge', idx+1, (u, v, t)))  
    sorted_components = sorted(unified, key=lambda x: x[2] if x[0]=='node' else x[2][2])

    def apply_mask(failed_nodes, failed_edges):
        nodes_masked = []
        for u, t in new_node_pairs:
            if u in failed_nodes:
                nodes_masked.append((u, 0.0))
            else:
                nodes_masked.append((u, t))
        edges_masked = []
        for idx, (u, v, t) in enumerate(new_edge_pairs, start=1):
            if idx in failed_edges:
                edges_masked.append((u, v, 0.0))
            else:
                edges_masked.append((u, v, t))
        return HKCC(edges_masked, nodes_masked, terminals)

    lo, hi = 0, len(sorted_components)
    flip_k = None
    while lo < hi:
        mid = (lo + hi) // 2
        failed_nodes = {cid for typ, cid, val in sorted_components[:mid] if typ=='node'}
        failed_edges = {cid for typ, cid, val in sorted_components[:mid] if typ=='edge'}
        if apply_mask(failed_nodes, failed_edges):
            lo = mid + 1
        else:
            flip_k = mid
            hi = mid
    if flip_k is None:
        return (-1, 0)

    typ, cid, val = sorted_components[flip_k-1]
    if typ == 'node':
        return (cid, val)
    else:
        return [cid, val]


def Component_state_vectors(new_nodes, new_edges):
    all_pairs = list(new_nodes) + [(f"e{idx}", t) for idx, (_, _, t) in enumerate(new_edges)]
    weights = np.array([pair[1] for pair in all_pairs])

    sorted_pairs = sorted(all_pairs, key=lambda x: x[1])
    sorted_weights = [pair[1] for pair in sorted_pairs if pair[1] != float('inf')]

    max_element = max(sorted_weights) + 100 if sorted_weights else 100
    sorted_weights.append(max_element)

    m = len(sorted_weights)  
    n = len(weights)         
    component_state_vectors = np.zeros((m, n), dtype=int)

    for i, s in enumerate(sorted_weights):
        component_state_vectors[i] = (weights >= s).astype(int)

    return component_state_vectors


def Generate_data(edge_pairs, node_pairs, terminals, label):
    all_component_state_vectors = []
    all_structure_function_value = []

    for new_node_pairs, new_edge_pairs in zip(node_pairs, edge_pairs):
        component_state_vectors = Component_state_vectors(new_node_pairs, new_edge_pairs)
        all_component_state_vectors.append(component_state_vectors)
        
        if label == 'train':
            k_terminal_pair = K_terminal_pair(new_edge_pairs, new_node_pairs, terminals)

            if isinstance(k_terminal_pair, tuple):
                node_id = k_terminal_pair[0]
                k_index = node_id - 1  
            else:
                edge_index = k_terminal_pair[0]
                num_nodes = len(new_node_pairs)
                k_index = num_nodes + (edge_index - 1) 

            structure_function_value = component_state_vectors[:, k_index].reshape(-1, 1)
            all_structure_function_value.append(structure_function_value)
            
    x_data = np.vstack(all_component_state_vectors)
    
    if label == 'train':
        y_data = np.vstack(all_structure_function_value)
        data = [x_data, y_data]
    else:
        data = [x_data]
    return data


def Generate_y(edge_pairs, node_pairs, terminals, Component_state_X, M):
    all_structure_function_value = []

    for index, (new_node_pairs, new_edge_pairs) in enumerate(zip(node_pairs, edge_pairs)):
        component_state_vectors = Component_state_X[index*M:(index+1)*M, ]

        k_terminal_pair = K_terminal_pair(new_edge_pairs, new_node_pairs, terminals)

        if isinstance(k_terminal_pair, tuple):
            node_id = k_terminal_pair[0]
            k_index = node_id - 1
        else:
            edge_index = k_terminal_pair[0]
            num_nodes = len(new_node_pairs)
            k_index = num_nodes + (edge_index - 1)

        structure_function_value = component_state_vectors[:, k_index].reshape(-1, 1)
        all_structure_function_value.append(structure_function_value)

    y_data = np.vstack(all_structure_function_value)
    return y_data


def survival_signature(ls, l_count, y_train):
    ls_index = np.all(l_count == ls, axis=1)
    ls_num = np.sum(ls_index)
    ls_surv_num = np.sum(y_train[ls_index].flatten())
    if ls_num==0:
        ls_phi=0
    else:
        ls_phi = ls_surv_num/ls_num
    return ls_phi


def prop(ls, S, N, F, t_value):
    p=1
    for s in range(S):
        m = N[s]
        l = ls[s]
        f = F[s]
        combi= math.comb(m, l)
        p=p*combi*(f(t_value)**(m-l))*((1-f(t_value))**l)
    return p


def build_l_count(x, groups):
    return np.stack([x[:, g].sum(axis=1) for g in groups], axis=1)


