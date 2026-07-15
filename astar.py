import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0 # Cost from start to current node
        self.h = 0 # Heuristic cost from current node to end
        self.f = 0 # Total cost
        
    def __eq__(self, other):
        return self.position == other.position
        
    def __lt__(self, other):
        return self.f < other.f

def astar(maze, start, end, dynamic_obstacles=None):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    maze: 2D array of strings ('0' = wall, '1' = path, '2' = target)
    start, end: tuples (r, c)
    """
    if dynamic_obstacles is None:
        dynamic_obstacles = []
        
    start_node = Node(start, None)
    end_node = Node(end, None)
    
    open_list = []
    closed_list = set()
    dyn_set = set(dynamic_obstacles)
    
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path
            
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            
            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[0]) -1) or node_position[1] < 0:
                continue
                
            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == '0':
                continue
                
            if node_position in dyn_set:
                continue
                
            new_node = Node(node_position, current_node)
            children.append(new_node)
            
        for child in children:
            if child.position in closed_list:
                continue
                
            child.g = current_node.g + 1
            # Manhattan distance heuristic
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h
            
            # Child is already in the open list
            # We don't bother checking if it's there with a lower cost for simplicity here,
            # as it works well enough for simple mazes
            heapq.heappush(open_list, child)
            
    return None # No path found
