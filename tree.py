
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
