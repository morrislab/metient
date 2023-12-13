repo_dir = os.path.join(os.getcwd(), "../../")

def read_lines(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()  # Yield each line, removing leading/trailing whitespace

def write_to_file(fn, data):
    with open(os.path.join(fn, fn), 'w') as f:
        for line in data:
            f.write(line)
            f.write("\n")
            
for site in ["m5", "m8"]:
    tree_fns = os.listdir(os.path.join(repo_dir, f"src/data/machina_sims/{site}_mut_trees"))
    for tree_fn in tree_fns:
        mut_trees_fn = os.path.join(repo_dir, f"src/data/machina_sims/{site}_mut_trees", tree_fn)
        out_fn = os.path.join(repo_dir, f"src/data/machina_sims/{site}_split_mut_trees")
        mig_pattern = tree_fn.split("_")[2]
        
        seed = tree_fn.split("_")[3].replace("seed", "").replace(".txt", "")
        with open(mut_trees_fn) as f:
            
            num_lines = sum([1 for line in f])
            print(num_lines)
            x = 0
            gen = read_lines(mut_trees_fn)
            print(mut_trees_fn)
            mut_line = next(gen)
            num_trees = next(gen).split()[0]
            print(num_trees, "trees")
            tree_line = next(gen)
            header = [mut_line, "1 #trees", tree_line]
            tree_data = [mut_line, "1 #trees", tree_line]
            tree_num = 0
            while x < num_lines - 3:
                same_tree = True
                while same_tree:
                    line = next(gen)
                    x += 1
                    # This marks the beginning of a tree
                    if "#edges, tree" in line:
                        same_tree = False
                        write_to_file(os.path.join(out_fn, f"mut_trees_{mig_pattern}_seed{seed}_tree{tree_num}.txt"), tree_data)
                        tree_num += 1
                        tree_data = [mut_line, "1 #trees", tree_line]
                    elif x == num_lines-3:
                        tree_data.append(line)
                        write_to_file(os.path.join(out_fn, f"mut_trees_{mig_pattern}_seed{seed}_tree{tree_num}.txt"), tree_data)
                        tree_num += 1
                        break
                    else:
                        tree_data.append(line)
                    
        print(tree_fn)
      
