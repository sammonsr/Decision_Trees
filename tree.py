MAX_DEPTH = 4


class Node: pass


class Leaf(Node):
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.index_in_parent = -1

    def __str__(self, level=0, label_dict=None):
        return "\t" * level + "- Leaf [Value: {}]".format(self.get_value_from_dict(label_dict)) + "\n"

    def get_value_from_dict(self, label_dict):
        return label_dict[self.value].decode('utf-8')


class Intermediate(Node):
    def __init__(self, children, attr_index):
        self.branch_conditions = []
        self.children = children
        self.attr_index = attr_index
        self.parent = None
        self.entropy = -1
        self.index_in_parent = -1
        self.class_dist = None

    def __str__(self, level=0, label_dict=None):
        if level == MAX_DEPTH:
            return ""

        ret = ("\t" * level) + "+ IntermediateNode [Column: {}, Entropy: {:.2f}, Conditions: {}, Distribution: {}]".format(
            self.attr_index, self.entropy, list(map(lambda a: a.condition_str, self.branch_conditions)),
            self.class_dist).replace("b'", "'") + "\n"

        for i, child in enumerate(self.children):
            ret +=  child.__str__(level=level + 1, label_dict=label_dict)
        return ret

    # Changing child inplace maintains original condition
    def replace_child(self, index, new_child):
        new_child.parent = self
        new_child.index_in_parent = index
        new_child.old_value = None
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

    def get_num_leafs(self, root=None, depth=1, max_depth=9999):
        if root is None:
            root = self

        if depth > max_depth:
            return 0

        if type(root) is Leaf:
            return 1

        num_leafs = 0
        for child in root.children:
            num_leafs = num_leafs + self.get_num_leafs(child, depth=depth + 1, max_depth=max_depth)

        return num_leafs
