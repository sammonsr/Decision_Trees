
class Node: pass

class Leaf(Node):
    def __init__(self, value):
        self.value = value


class Intermediate(Node):
    def __init__(self, condition, children):
        self.condition = condition
        self.children = children

    def add_child(self, child):
        self.children.append(child)

    def pass_condition(self, value):
        return self.condition(value)
