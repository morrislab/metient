def get_adj_matrices_from_spruce_mutation_trees(mut_trees_filename, idx_to_character_label, is_sim_data=False):
    '''
    When running MACHINA's generatemutationtrees executable (SPRUCE), it provides a txt file with
    all possible mutation trees. See data/machina_simulated_data/mut_trees_m5/ for examples

    Returns a list of tuples, each containing (T, pruned_idx_to_character_label) for each
    tree in mut_trees_filename.
        - T: adjacency matrix where Tij = 1 if there is a path from i to j
        - idx_to_character_label: a dict mapping indices of the adj matrix T to character
        labels 
    '''
    out = []
    with open(mut_trees_filename, 'r') as f:
        tree_data = []
        for i, line in enumerate(f):
            if i < 3: continue
            # This marks the beginning of a tree
            if "#edges, tree" in line:
                adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
                out.append((adj_matrix, pruned_idx_to_label))
                tree_data = []
            else:
                nodes = line.strip().split()
                # fixes incompatibility with naming b/w cluster file (uses ";" separator)
                # and the ./generatemutationtrees output (uses "_" separator)
                if is_sim_data:
                    tree_data.append((";".join(nodes[0].split("_")), ";".join(nodes[1].split("_"))))
                else:
                    tree_data.append((nodes[0], nodes[1]))

        adj_matrix, pruned_idx_to_label = _get_adj_matrix_from_machina_tree(tree_data, idx_to_character_label)
        out.append((adj_matrix, pruned_idx_to_label))
    return out