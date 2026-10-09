#!/usr/bin/env python
# coding: utf-8

# # Search: Solving a Maze Using a Goal-based Agent
# 
# Student Name: [Add your name]
# 
# I have used the following AI tools: [list tools]
# 
# I understand that my submission needs to be my own work: [your initials]

# ## Learning Outcomes
# 
# * Formulate search problems using key components like initial state, actions, and goal state in a deterministic, fully observable environment.
# * Implement and compare search algorithms including BFS, DFS, GBFS, A*, and IDS for planning paths through mazes.
# * Analyze algorithm performance by measuring path cost, node expansions, depth, and memory usage across various maze types.
# * Use visualization tools to represent maze paths and support debugging and analysis.

# ## Instructions
# 
# Total Points: Undergrads 100 + 5 bonus / Graduate students 110
# 
# Complete this notebook. Use the provided notebook cells and insert additional code and markdown cells as needed. Submit the notebook file and the completely rendered notebook with all outputs as a HTML file. 
# 
# 
# ## Introduction
# 
# In this exercise, we will implement the planning function for a type of goal-based agent called a __planning agent__. The planning function uses a map it is given to plan a path through the maze from the starting location $S$ to the goal location $G$. We will only focus on the planning function, so you do not need to implement an environment, just use the map to search for a path to solve the maze. 
# 
# Once the plan is made, the agent in a deterministic environment (i.e., the transition function is deterministic with the outcome of each state/action pair fixed and no randomness) can just follow the plan step-by-step and does not need to care about the percepts.
# This is also called an **[open-loop system](https://en.wikipedia.org/wiki/Open-loop_controller).**
# The execution phase is trivial and can be executed using a model-based reflex agent 
# that ignores all percepts and just follows the plan. I will show you a short example, but you do not implement it in this exercise.
# 
# Given that the agent has a complete and correct map, the environment is **fully observable, discrete, deterministic, and known.** 
# Remember:
# 
# * **Fully observable** means that the agent can see its state and what the available actions are. That means the **percepts contain the complete current state.**
# Here, during planning, the agent always sees its x and y coordinates on the map and
# also seeks when it has reached the goal state. 
# * **Discrete** means that we have a **finite set of states.** The maze has a finite set 
# of squares the agent can be in.
# * **Deterministic** means that the **transition function contains no randomness.** An action in a state will always produce the same result. Going south from the start state always will lead to the same square.
# * **Know** means that the agent **knows the complete transition function.** The 
# agent has the map and therefore knows how its position changes when it walks in a direction.
# 
# Tree search algorithm implementations that you find online typically come from data structures courses and have a different aim than AI tree search. These algorithms assume that you already have a tree in memory. We are interested in dynamically creating a search tree with the aim of finding a good/the best path from the root note to the goal state. Follow the pseudo code presented in the text book (and replicated in the slides) closely. Ideally, we would like to search only a small part of the maze, i.e., create a search tree with as few nodes as possible. 
# 
# Several mazes for this exercise are stored as text files. We need to make sure that they are locally available.

# In[4]:


# change directory to Google Drive for Google Colab
try:
    from google.colab import drive
    import os

    drive.mount('/content/drive')
    os.chdir('/content/drive/My Drive/Colab Notebooks/')
    print("Your working directory is now Google Drive: My Drive > Colab Notebooks")
except ImportError:
    pass


# In[ ]:


import urllib.request
import os

def download(file, base_url):
    if not os.path.exists(file):
        urllib.request.urlretrieve(base_url + file, file)

base_url = "https://raw.githubusercontent.com/mhahsler/Introduction_to_Artificial_Intelligence/refs/heads/master/Search/"
download("maze_helper.py", base_url)

mazes = ["small_maze.txt", "medium_maze.txt", "large_maze.txt", "L_maze.txt", "empty_maze.txt", "empty_maze_2.txt", "loops_maze.txt", "open_maze.txt", ]
for maze in mazes:
    download(maze, base_url)


# Here is the small example maze:

# In[4]:


with open("small_maze.txt", "r") as f:
    maze_str = f.read()
print(maze_str)


# **Note:** If you get an error here that the file cannot be found, then you need to download it. See [HOWTO Work on Assignments.](https://github.com/mhahsler/CS7320-AI/blob/master/HOWTOs/working_on_assignments.md)

# ## Parsing and pretty printing the maze
# 
# The maze can also be displayed in color using code in the module [maze_helper.py](maze_helper.py). The code parses the string representing the maze and converts it into a `numpy` 2d array which you can use in your implementation. Position are represented as a 2-tuple of the form `(row, col)`. 

# In[5]:


import maze_helper as mh

maze = mh.parse_maze(maze_str)

# look at a position in the maze by subsetting the 2d array
print("Position(0,0):", maze[0, 0])

# there is also a helper function called `look(maze, pos)` available
# which uses a 2-tuple for the position.
print("Position(8,1):", mh.look(maze, (8, 1)))


# A helper function to visualize the maze is also available.

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# use higher resolution images in notebooks

mh.show_maze(maze)


# Find the `(x,y)` position of the start and the goal using the helper function `find_pos()`

# In[7]:


print("Start location:", mh.find_pos(maze, what = "S"))
print("Goal location:", mh.find_pos(maze, what = "G"))


# Helper function documentation.

# In[8]:


help(mh)


# You will need to make a local copy of the module file [maze_helper.py](maze_helper.py) in the same folder where your notebook is.

# ## An Example for a Planning Agent
# 
# I will show you here how to implement a simple agent that uses a random plan. It will not solve the maze, but show you how the mechanics work.
# 
# First, we define a generic planning agent that fist plans, and then executes the plan step-by-step. 

# In[9]:


class Planning_Agent:
    def __init__(self, maze, start, goal, planning_function):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.planning_function = planning_function
        self.plan = None
        self.progress = None

    def act(self):
        # plan if no plan exists
        if self.plan is None:
            print("Planning...")
            self.plan = self.planning_function(self.maze, self.start, self.goal)
            self.progress = 0

        # check if plan is completed
        if self.progress >= len(self.plan):        
            raise Exception("Completed Plan. No more planned actions")

        # follow the plan
        action = self.plan[self.progress]
        print(f"Following plan... step {self.progress}: {action}")

        self.progress += 1
        return action


# Next, we define the planning function. This function is what you will implement in this assignment.  

# In[10]:


import numpy as np

def plan_random(maze, start, goal):
    """Create a random plan with 10 steps"""
    plan = np.random.choice(["N", "E", "S", "W"], size=10, replace=True).tolist()
    return plan

plan_random(maze, (1,1), (8,8))


# This planning function is not great and will not produce a plan that solves the maze. Your planning functions will do better.
# 
# Finally, we can create the planning agent, give it the planning function and implement a simple environment that asks it 11 times for an action.

# In[11]:


my_agent = Planning_Agent(maze, mh.find_pos(maze, what = "S"), mh.find_pos(maze, what = "G"), plan_random)

def environment(agent_function, steps):
    for _ in range(steps):
        try:
            agent_function()
        except Exception as e:
            print(f"Agent exception: {e}")

environment(my_agent.act, steps=11)


# Note: The agent and environment implementation above is just an illustration. You will only implement and experiment with different versions of the planning function.

# ## Tree structure
# 
# To use tree search, you will need to implement a tree data structure in Python. 
# Here is an implementation of the basic node structure for the search algorithms (see Fig 3.7 on page 73). I have added a method that extracts the path from the root node to the current node. It can be used to get the path when the search is completed.

# In[12]:


class Node:
    def __init__(self, pos, parent, action, cost):
        self.pos = tuple(pos)    # the state; positions are (row,col)
        self.parent = parent     # reference to parent node. None means root node.
        self.action = action     # action used in the transition function (root node has None)
        self.cost = cost         # for uniform cost this is the depth. It is also g(n) for A* search

    def __str__(self):
        return f"Node - pos = {self.pos}; action = {self.action}; cost = {self.cost}"

    def get_path_from_root(self):
        """returns nodes on the path from the root to the current node."""
        node = self
        path = [node]

        while not node.parent is None:
            node = node.parent
            path.append(node)

        path.reverse()

        return(path)


# If needed, then you can add more fields to the class like the heuristic value $h(n)$ or $f(n)$.
# 
# Examples for how to create and use a tree and information on memory management can be found [here](../HOWTOs/trees.ipynb).

# # Tasks
# 
# The goal is to:
# 
# 1. Implement the following search algorithms for solving different mazes:
# 
#     - Breadth-first search (BFS)
#     - Depth-first search (DFS)
#     - Greedy best-first search (GBFS)
#     - A* search
# 
# 2. Run each of the above algorithms on the 
#     - [small maze](small_maze.txt), 
#     - [medium maze](medium_maze.txt), 
#     - [large maze](large_maze.txt), 
#     - [open maze](open_maze.txt),
#     - [L maze](L_maze.txt),
#     - [loops maze](loops_maze.txt),
#     - [empty maze](empty_maze.txt), and
#     - [empty maze (rotated)](empty_maze_2.txt).
#     
# 3. For each problem instance and each search algorithm, report the following in a table:
# 
#     - The solution and its path cost
#     - Total number of nodes expanded
#     - Maximum tree depth
#     - Maximum size of the frontier
# 
# 4. Display each solution by marking every maze square (or state) visited and the squares on the final path.
# 
# ## General [10 Points]
# 
# 1. Make sure that you use the latest version of this notebook.
# 2. Your implementation can use libraries like math, numpy, scipy, but not libraries that implement intelligent agents or complete search algorithms. Try to keep the code simple! In this course, we want to learn about the algorithms and we often do not need to use object-oriented design.
# 3. You notebook needs to be formatted professionally. 
#     - Add additional markdown blocks for your description, comments in the code, add tables and use mathplotlib to produce charts where appropriate
#     - Do not show debugging output or include an excessive amount of output.
#     - Check that your submitted file is readable and contains all figures.
# 4. Document your code. Use comments in the code and add a discussion of how your implementation works and your design choices.

# ## Task 1: Defining the search problem and determining the problem size [10 Points]
# 
# Define the components of the search problem:
# 
# * Initial state
# * Actions
# * Transition model
# * Goal state
# * Path cost
# 
# Use verbal descriptions, variables and equations as appropriate. 
# 
# *Note:* You can switch the next block from code to Markdown and use formatting.

# In[13]:


# Your answer goes here


# Give some estimates for the problem size:
# 
# * $n$: state space size
# * $d$: depth of the optimal solution
# * $m$: maximum depth of tree
# * $b$: maximum branching factor
# 
# Describe how you would determine these values for a given maze.

# In[14]:


# Your answer goes here


# ## Task 2: Uninformed search: Breadth-first and depth-first [40 Points]
# 
# Implement these search strategies. Follow the pseudocode in the textbook/slides. You can use the tree structure shown above to extract the final path from your solution.
# 
# Read the following **important notes** carefully:
# * You can find maze solving implementations online that use the map to store information. While this is an effective idea for this two-dimensional navigation problem, it typically cannot be used for other search problems. Therefore, follow the textbook and **do not store information in the map.** Only store information in the tree created during search, and use the `reached` and `frontier` data structures where appropriate.
# * DSF behavior can be implemented using the BFS tree search algorithm and simply changing the order in which the frontier is expanded (this is equivalent to best-first search with path length as the criterion to expand the next node). However, this would be a big mistake since it combines the bad space complexity of BFS with the bad time complexity of DFS! **To take advantage of the significantly smaller memory footprint of DFS, you need to implement DFS in a different way without a `reached` data structure (often also called `visited` or `explored`) and by releasing the memory for nodes that are not needed anymore.**
# * Since the proper implementation of DFS does not use a `reached` data structure, redundant path checking abilities are limited to cycle checking. 
# You need to implement **cycle checking since DSF is incomplete (produces an infinite loop) if cycles cannot be prevented.** You will see in your experiments that cycle checking in open spaces is challenging.

# In[15]:


# Your code goes here


# How does BFS and DFS (without a reached data structure) deal with loops (cycles)?

# In[16]:


# Discussion


# Are your implementations complete and optimal? Explain why. What is the time and space complexity of each of **your** implementations? Especially discuss the difference in space complexity between BFS and DFS.

# In[17]:


# Discussion


# ## Task 3: Informed search: Implement greedy best-first search and A* search  [20 Points]
# 
# You can use the map to estimate the distance from your current position to the goal using the Manhattan distance (see https://en.wikipedia.org/wiki/Taxicab_geometry) as a heuristic function. Both algorithms are based on Best-First search which requires only a small change from the BFS algorithm you have already implemented (see textbook/slides). 

# In[18]:


# Your code goes here


# Are your implementations complete and optimal? What is the time and space complexity?

# In[19]:


# Discussion


# ## Task 4: Comparison and discussion [20 Points] 
# 
# Run experiments to compare the implemented algorithms.
# 
# How to deal with issues:
# 
# * Your implementation returns unexpected results: Try to debug and fix the code. Visualizing the maze, the current path and the frontier after every step is very helpful. If the code still does not work, then mark the result with an asterisk (*) and describe the issue below the table.
# 
# * Your implementation cannot consistently solve a specific maze and ends up in an infinite loop:
#     Debug (likely your frontier and cycle checking for DFS are the issue). If it is a shortcoming of the algorithm/implementation, then put "N/A*" in the results table and describe why this is happening.

# In[20]:


# Add code


# Complete the following table for each maze.
# 
# __Small maze__
# 
# | algorithm | path cost | # of nodes expanded | max tree depth | max # of nodes in memory | max frontier size |
# |-----------|-----------|----------------|----------------|---------------|-------------------|
# | BFS       |           |                |                |               |                   |
# | DFS       |           |                |                |               |                   |
# | GBS       |           |                |                |               |                   |
# | A*        |           |                |                |               |                   |
# 
# __Medium Maze__
# 
# ...

# Present the results as using charts (see [Python Code Examples/charts and tables](../HOWTOs/charts_and_tables.ipynb)). 

# In[21]:


# Add charts


# Discuss the most important lessons you have learned from implementing the different search strategies. 

# In[22]:


# Add discussion


# ## Advanced task: IDS and Multiple goals
# 
# * __Graduate students__ need to complete this task [10 points]
# * __Undergraduate students__ can attempt this as a bonus task [max +5 bonus points].
# 
# ### IDS 
# Implement IDS (iterative deepening search) using your DFS implementation. Test IDS on the mazes above. You may run into some issues with mazes with open spaces. If you cannot resolve the issues, then report and discuss what causes the problems.

# In[23]:


# Your code/answer goes here


# ### Multiple Goals 
# Create a few mazes with multiple goals by adding one or two more goals to the medium size maze. The agent is done when it finds one of the goals.
# Solve the maze with your implementations for DFS, BFS, and IDS. Run experiments to show which implementations find the optimal solution and which do not. Discuss why that is the case.

# In[24]:


# Your code/answer goes here


# ## More Advanced Problems to Think About (not for credit)
# 
# If the assignment was to easy for yuo then you can think about the following problems. These problems are challenging and not part of this assignment. 
# 
# ### Intersection as States
# Instead of defining each square as a state, use only intersections as states. Now the storage requirement is reduced, but the path length between two intersections can be different. If we use total path length measured as the number of squares as path cost, how can we make sure that BFS and iterative deepening search is optimal? Change the code to do so.

# In[25]:


# Your code/answer goes here


# ### Weighted A* search
# Modify your A* search to add weights (see text book) and explore how different weights influence the result.

# In[26]:


# Your code/answer goes here


# ### Unknown Maze
# What happens if the agent does not know the layout of the maze in advance? This means that the agent faces an unknown environment, where it does not know the transition function. How does the environment look then (PEAS description)? How would you implement a rational agent to solve the maze? What if the agent still has a GPS device to tell the distance to the goal?

# In[27]:


# Your code/answer goes here

