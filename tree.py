
class Node: pass


class Leaf(Node):
    def __init__(self, value):
        self.value = value


class Intermediate(Node):
    def __init__(self, children, attr_index):
        self.branch_conditions = []
        self.children = children
        self.attr_index = attr_index
        self.parent = None
        self.index_in_parent = -1

    # Changing child inplace maintains original condition
    def replace_child(self, index, new_child):
        self.children[index] = new_child

    def add_child(self, child, condition):
        self.children.append(child)
        self.branch_conditions.append(condition)
        child.parent = self

        child.index_in_parent = len(self.children) - 1

    def get_depth(self, root=None):
        if root is None:
            root = self

        if type(root) is Leaf:
            return 1


        max_child_depth = -1
        for child in root.children:
            max_child_depth = max(max_child_depth, self.get_depth(child))


        return max_child_depth + 1

    def get_num_leafs(self, root=None):
        if root is None:
            root = self

        if type(root) is Leaf:
            return 1

        num_leafs = 0
        for child in root.children:
            num_leafs = num_leafs + self.get_num_leafs(child)

        return num_leafs
