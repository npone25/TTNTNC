import nbformat

def update_notebook():
    with open('Ex1_Maze.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # The code to add
    code = """# Implementation of Search Algorithms
import numpy as np

def get_successors(maze, pos):
    successors = []
    directions = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    for action, (dr, dc) in directions.items():
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1]:
            # check if not wall (assuming walls are 'X' or represented in some way)
            if str(maze[r, c]).strip() != 'X' and str(maze[r, c]).strip() != '1':
                successors.append((action, (r, c), 1))
    return successors

def plan_bfs(maze, start, goal):
    start_node = Node(start, None, None, 0)
    if start == goal: return []
    frontier = [start_node]
    reached = {tuple(start)}
    
    while frontier:
        node = frontier.pop(0)
        if node.pos == goal:
            return [n.action for n in node.get_path_from_root()[1:]]
            
        for action, next_pos, step_cost in get_successors(maze, node.pos):
            if next_pos not in reached:
                reached.add(next_pos)
                child = Node(next_pos, node, action, node.cost + step_cost)
                frontier.append(child)
    return None

def plan_dfs(maze, start, goal):
    start_node = Node(start, None, None, 0)
    frontier = [start_node]
    
    while frontier:
        node = frontier.pop()
        if node.pos == goal:
            return [n.action for n in node.get_path_from_root()[1:]]
            
        path_positions = {n.pos for n in node.get_path_from_root()}
        
        for action, next_pos, step_cost in get_successors(maze, node.pos):
            if next_pos not in path_positions:
                child = Node(next_pos, node, action, node.cost + step_cost)
                frontier.append(child)
    return None

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def plan_gbfs(maze, start, goal):
    start_node = Node(start, None, None, 0)
    setattr(start_node, 'h_cost', manhattan(start, goal))
    frontier = [start_node]
    reached = {tuple(start)}
    
    while frontier:
        frontier.sort(key=lambda n: n.h_cost)
        node = frontier.pop(0)
        
        if node.pos == goal:
            return [n.action for n in node.get_path_from_root()[1:]]
            
        for action, next_pos, step_cost in get_successors(maze, node.pos):
            if next_pos not in reached:
                reached.add(next_pos)
                child = Node(next_pos, node, action, node.cost + step_cost)
                setattr(child, 'h_cost', manhattan(next_pos, goal))
                frontier.append(child)
    return None

def plan_astar(maze, start, goal):
    start_node = Node(start, None, None, 0)
    setattr(start_node, 'h_cost', manhattan(start, goal))
    frontier = [start_node]
    reached = {tuple(start): start_node.cost}
    
    while frontier:
        frontier.sort(key=lambda n: n.cost + n.h_cost)
        node = frontier.pop(0)
        
        if node.pos == goal:
            return [n.action for n in node.get_path_from_root()[1:]]
            
        for action, next_pos, step_cost in get_successors(maze, node.pos):
            new_cost = node.cost + step_cost
            if next_pos not in reached or new_cost < reached[next_pos]:
                reached[next_pos] = new_cost
                child = Node(next_pos, node, action, new_cost)
                setattr(child, 'h_cost', manhattan(next_pos, goal))
                frontier.append(child)
    return None

def plan_dls(maze, start, goal, limit):
    start_node = Node(start, None, None, 0)
    frontier = [start_node]
    
    while frontier:
        node = frontier.pop()
        if node.pos == goal:
            return [n.action for n in node.get_path_from_root()[1:]]
            
        if node.cost < limit:
            path_positions = {n.pos for n in node.get_path_from_root()}
            for action, next_pos, step_cost in get_successors(maze, node.pos):
                if next_pos not in path_positions:
                    child = Node(next_pos, node, action, node.cost + step_cost)
                    frontier.append(child)
    return "CUTOFF"

def plan_ids(maze, start, goal):
    limit = 0
    while limit < 1000:
        result = plan_dls(maze, start, goal, limit)
        if result != "CUTOFF" and result is not None:
            return result
        limit += 1
    return None
"""

    new_cell = nbformat.v4.new_code_cell(source=code)
    nb.cells.append(new_cell)

    with open('Ex1_Maze.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    update_notebook()
