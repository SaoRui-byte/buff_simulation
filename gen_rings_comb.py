import itertools

file = open("rings_combinations.txt", "w")
# -1 means no ring, 0 means aiming, 1-11 means ring with that number

states = [-1,0] + list(range(1, 12))

all_combinations = list(itertools.product(states, repeat=5))
valid_combinations = [comb for comb in all_combinations if comb[0] == 0 and comb.count(0) == 1]

for combo in valid_combinations:
    file.write(str(combo) + "\n")
file.close()