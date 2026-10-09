#!/usr/bin/env python
# coding: utf-8

# # Comparing Different Search Strategies: Maze
# 
# Notes on breaking ties: 
# 
# * The order in which the children are explored (see `available directions`) makes a big difference for DFS and dealing with empty spaces. I explore the directions in random order which makes the algorithm stochastic!
# * Ties for $h(n)$ and $f(n)$ need to be broken in a consistent manner. I use the most recently added node. To try to keep moving into the same direction.
# 
# 

# In[1]:


get_ipython().run_line_magic('run', 'maze_helper.py')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np

#f = open("small_maze.txt", "r")
#f = open("medium_maze.txt", "r")
#f = open("large_maze.txt", "r")    # this has only one solution!
#f = open("open_maze.txt", "r")
#f = open("empty_maze.txt", "r")
#f = open("empty_maze_2.txt", "r")
#f = open("loops_maze.txt", "r")
f = open("L_maze.txt", "r")

maze_str = f.read()
maze = parse_maze(maze_str)

# look at two positions in the maze
print("Position(0,0):", maze[0, 0])
print("Position(8,1):", maze[8, 1])

show_maze(maze)


# ## Implementation
# 
# My implementation follows the pseudo code from the slides/textbook.

# In[2]:


# tree_search_solution.py has my actual implementation (not published)
import tree_search_solution as ts


# order in which we add new states to the frontier
ts.set_order("NESW")
#ts.set_order(random=True)


# ## Experiments

# ### BFS

# In[3]:


ts.set_order("NESW")
#ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "BFS", debug = False, vis = False)')
ts.show_path(maze, result)
result['actions']


# ### DFS 
# 
# This implementation uses not reached data structure and has space complexity $O(bm)$ instead of $O(b^m)$ when we reuse the tree search algorithm from BFS!
# 
# We need to check for all cycles. If we do not break all cycles correctly, then we will end up in an infinite loop. Here are possible solutions:
# * Stop after a fixed number of tries and return no solution `max_tries`.
# * IDS solves this problem.
# 
# Note on the visualization: I use gray for areas that the algorithm has explored, but DFS has already removed it from memory!

# In[4]:


# Note: DFS uses LIFO so the directions come from the stack in reverse order!

# Can get stuck for empty maze since cycle checking is not string enough! 
# I use a maximum number of tries and stop if the goal is not reached.
ts.set_order("NESW")
#ts.set_order("SENW")
#ts.set_order("WSEN")
#ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.DFS(maze, vis = False, max_tries = 100000, debug_reached = True)')

#result
ts.show_path(maze, result)
if result['path'] is None:
    print("No solution found!")


# Do the same, but change the order in which we explore states (add them to them as nodes to the frontier).

# In[5]:


#ts.set_order("NESW")
ts.set_order("SENW")
#ts.set_order("WSEN")
#ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.DFS(maze, vis = False, max_tries = 100000, debug_reached = True)')

#result
ts.show_path(maze, result)
if result['path'] is None:
    print("No solution found!")


# We could do a random walk and not check for cycles. This is guaranteed to reach eventually any square including the goal, 
# but creates a long path. The path could be simplified leading to the [Tremaux's algorithm](https://en.wikipedia.org/wiki/Maze_solving_algorithm).

# In[6]:


ts.set_order(random = True)

# run it multiple times to see the effect of randomization
for _ in range(5):
    get_ipython().run_line_magic('time', 'result = ts.DFS(maze, check_cycle = False, max_tries = 100000, vis = False, debug_reached = True)')

    #result
    ts.show_path(maze, result)
    if result['path'] is None:
        print("No solution found!")


# ### Run randomized DFS multiple times and use the best solution.
# 
# __Note:__ IDS takes a similar amount of time and memory, but is guaranteed optimal.

# In[7]:


ts.set_order(random = True)

N = 100
get_ipython().run_line_magic('time', 'results = [ ts.DFS(maze, max_tries = 10000, vis = False) for _ in range(N) ]')

# check if we found a solution and display the best solution
results = [ r for r in results if not r['path'] is None ]
if len(results) > 0:
    path_lengths = [ len(r['path'])-1 for r in results ]

    print(f"Solutions have path_lengths of {path_lengths}")

    result = results[ts.min_index(path_lengths)]
    ts.show_path(maze, result)
else:
    print("No solution found!")


# ### Depth limited DFS
# 
# Note: The frontier needs to be checked differently during cycle checking!

# In[8]:


ts.set_order(random = True)

get_ipython().run_line_magic('time', 'result = ts.DFS(maze, limit = 5, frontier_option = 2, max_tries = 100000, vis = False, debug_reached = True)')
ts.show_path(maze, result)


# ### IDS
# 
# __Notes:__ 
# 
# * IDS with DFS does not store reached squares, so gray areas are not shown!
# 
# * IDS depends on the cycle checking of DFS and therefore is also affected by these problems.

# In[9]:


ts.set_order(random = True)

get_ipython().run_line_magic('time', 'result = ts.IDS(maze, frontier_option = 2, max_tries = 100000)')
ts.show_path(maze, result)


# ### Greedy Best-First Search (GBFS)

# In[10]:


# set the heuristic to Manhattan distance
ts.heuristic = ts.manhattan


# In[11]:


ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "GBFS", debug = False, vis = False)')
ts.show_path(maze, result)


# ### A* Search

# In[12]:


ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", debug = False, vis = False)')
ts.show_path(maze, result)


# ### Weighted A* Search

# $W > 1$ tends towards GBFS (optimality is not guaranteed)

# In[13]:


ts.set_order(random=True)

get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", W = 1+1e-9, debug = False, vis = False)')
ts.show_path(maze, result)


# In[14]:


get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", W = 5, debug = False, vis = False)')
ts.show_path(maze, result)


# In[15]:


get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", W = 1000, debug = False, vis = False)')
ts.show_path(maze, result)


# $W<1$ tends towards Uniform-Cost Search/BFS (optimality is guaranteed)

# In[16]:


get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", W = .7, debug = False, vis = False)')
ts.show_path(maze, result)


# In[17]:


get_ipython().run_line_magic('time', 'result = ts.best_first_search(maze, strategy = "A*", W = .0001, debug = False, vis = False)')
ts.show_path(maze, result)


# ### Compare Timing

# In[18]:


#f = open("small_maze.txt", "r")
#f = open("medium_maze.txt", "r")
f = open("large_maze.txt", "r")    # this has only one solution!
#f = open("open_maze.txt", "r")
#f = open("empty_maze.txt", "r")
#f = open("empty_maze_2.txt", "r")
#f = open("L_maze.txt", "r")
#f = open("loops_maze.txt", "r")

maze_str = f.read()
maze = parse_maze(maze_str)

ts.show_maze(maze)

ts.set_order(random=True)


# In[22]:


import timeit
import math

reps = 10

times = {}

algorithms = ["BFS", "DFS", "GBFS", "A*"]

### FIXME: add IDS

for a in algorithms:
    times[a] = math.ceil(timeit.timeit(stmt = f'ts.best_first_search(maze, strategy = "{a}", debug = False, vis = False)', 
              setup = 'from __main__ import ts, maze',
                    number = reps)*1e6/reps)

times['DFS(no reached)'] = math.ceil(timeit.timeit(stmt = f'ts.DFS(maze, vis = False)', 
              setup = 'from __main__ import ts, maze',
                    number = reps)*1e6/reps)

times['IDS'] = math.ceil(timeit.timeit(stmt = f'ts.IDS(maze, vis = False)', 
              setup = 'from __main__ import ts, maze',
                    number = reps)*1e6/reps)


# In[23]:


import pandas as pd
df = pd.DataFrame(times, index = ["time in micro seconds"])
df


# In[24]:


import matplotlib.pyplot as plt

plt.bar(df.columns, height = df.iloc[0])
plt.ylabel("run time in micro seconds")
plt.show()

