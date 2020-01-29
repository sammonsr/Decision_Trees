
class Node: pass


class Leaf(Node):
    def __init__(self, value):
        self.value = value


class Intermediate(Node):
    def __init__(self, children, attr_index):
        self.branch_conditions = []
        self.children = children
        self.attr_index = attr_index

    def add_child(self, child, condition):
        self.children.append(child)
        self.branch_conditions.append(condition)

    def pass_condition(self, value):
        return self.condition(value)

    def get_depth(self, root=None):
        if root is None:
            root = self

        if type(root) is Leaf:
            return 1


        max_child_depth = -1
        for child in self.children:
            max_child_depth = max(max_child_depth, self.get_depth(child))


        return max_child_depth + 1

    def get_num_leafs(self, root=None):
        if root is None:
            root = self

        if type(root) is Leaf:
            return 1

        num_leafs = 0

        for child in self.children:
            num_leafs = num_leafs + self.get_num_leafs(child)

        return num_leafs
