from heapq import heapify, heappush, heappushpop, nlargest

class MinHeap():
    def __init__(self, k):
        self.h = []
        self.length = k
        self.items = set()
        heapify(self.h)
        
    def add(self, loss, A, V, soft_V, i=0):
        # Maintain a max heap so that we can efficiently 
        # get rid of larger loss value Vs
        tree = vutil.LabeledTree(A, V)
        if (len(self.h) < self.length) and (tree not in self.items): 
            self.items.add(tree)
            heappush(self.h, VertexLabelingSolution(loss, V, soft_V, i))
        # If loss is greater than the max loss we
        # already have, don't bother adding this 
        # solution (hash checking below is expensive)
        elif loss > self.h[0].loss:
            return
        # If we've reached capacity, push the new
        # item and pop off the max item
        elif tree not in self.items:
            self.items.add(tree)
            removed = heappushpop(self.h, VertexLabelingSolution(loss, V, soft_V, i))
            removed_tree = vutil.LabeledTree(A, removed.V)
            self.items.remove(removed_tree)
        
    def get_top(self):
        # due to override in comparison operator, this
        # actually returns the n smallest values
        return nlargest(self.length, self.h)


# mig_vec = get_mig_weight_vector(batch_size, input_weights)
# seed_vec = get_seed_site_weight_vector(batch_size, input_weights)
# for sln in final_solutions:
#     print(sln.loss, sln.i, mig_vec[sln.i], seed_vec[sln.i])
# with open(os.path.join(output_dir, f"{run_name}.txt"), 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(file_output)
# with open(os.path.join(output_dir, f"{run_name}_best_weights.txt"), 'w', newline='') as file:
#     file.write(f"{best_pars_weights[0]}, {best_pars_weights[1]}")
