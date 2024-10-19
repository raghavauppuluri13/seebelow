import numpy as np


class Node:
    def __init__(self, x, y, width, height):
        self.x = x  # X-coordinate of the top-left corner
        self.y = y  # Y-coordinate of the top-left corner
        self.width = width
        self.height = height
        self.children = []  # Will have 4 children when subdivided
        self.points = []  # This node's points
        self.isLeaf = True

    def subdivide(self):
        # Create four children
        half_width = self.width // 2
        half_height = self.height // 2
        self.children = [
            Node(self.x, self.y, half_width, half_height),
            Node(self.x + half_width, self.y, half_width, half_height),
            Node(self.x, self.y + half_height, half_width, half_height),
            Node(self.x + half_width, self.y + half_height, half_width, half_height),
        ]
        self.isLeaf = False


class QuadTree:
    def __init__(self, width, height, min_width, min_height):
        self.root = Node(0, 0, width, height)
        self.min_width = min_width
        self.min_height = min_height
        self.groups = {}  # Mapping for O(1) retrieval

    def _get_group_key(self, point):
        # Create a unique key based on the group's top-left corner to which the point belongs
        group_x = (point[0] // self.min_width) * self.min_width
        group_y = (point[1] // self.min_height) * self.min_height
        return (group_x, group_y)

    def insert(self, point, X_idx):
        # Wrapper function to insert a point and update the dictionary
        self._insert(self.root, point)
        group_key = self._get_group_key(point)
        if group_key not in self.groups:
            self.groups[group_key] = []
        self.groups[group_key].append(X_idx)

    def _insert(self, node, point):
        # Recursively insert a point into the QuadTree
        if node.isLeaf:
            # If the node is too large, subdivide it
            if node.width > self.min_width or node.height > self.min_height:
                node.subdivide()
                # Reinsert the node's points into the new children
                for p in node.points:
                    self._insert_into_children(node, p)
                node.points = []  # Clear the points since they've been redistributed
                self._insert_into_children(node, point)  # Insert the new point
            else:
                node.points.append(
                    point
                )  # If the node is small enough, store the point
        else:
            # Not a leaf, so insert into the appropriate child
            self._insert_into_children(node, point)

    def _insert_into_children(self, node, point):
        # Insert a point into the appropriate child of a node
        for child in node.children:
            if (
                child.x <= point[0] < child.x + child.width
                and child.y <= point[1] < child.y + child.height
            ):
                self._insert(child, point)
                break

    @property
    def group_area(self):
        return self.min_width * self.min_height

    def get_group_dict(self):
        # Get the entire dictionary of groups
        return self.groups


if __name__ == "__main__":
    x = np.linspace(0, 100, 5)
    y = np.linspace(0, 100, 5)
    grid = np.meshgrid(x, y)

    grid = np.concatenate([grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)], axis=1)
    print(grid.shape)

    quad_tree = QuadTree(100, 100, 10, 10)

    for point in grid:
        point = tuple(point)
        quad_tree.insert(point, 0)
    print(len(quad_tree.get_group_dict().keys()))  # Retrieve the entire dictionary
