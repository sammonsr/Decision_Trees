import matplotlib.pyplot as plt

from tree import Leaf, Intermediate

NODE_TEXT = "*"

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

class Visualiser:
    def __init__(self, tree, label_dict):
        self.whole_tree = tree
        self.label_dict = label_dict
        self.tree_width = tree.get_num_leafs()
        self.tree_depth = tree.get_depth()
        self.x_off = -0.5 / self.tree_width
        self.y_off = 1

        self.ax1 = None

    def plot_tree(self, parent_point, node_txt, tree=None):
        if tree is None:
            tree = self.whole_tree

        num_leafs = tree.get_num_leafs()

        center_point = (self.x_off + (1.0 + float(num_leafs)) / 2.0 / self.tree_width, self.y_off)

        self.plot_mid_text(center_point, parent_point, node_txt)
        self.plot_node(NODE_TEXT, center_point, parent_point, decision_node)

        self.y_off = self.y_off - 1 / self.tree_depth

        for i in range(len(tree.children)):
            child = tree.children[i]
            if type(child) is Intermediate:
                self.plot_tree(center_point, tree.branch_conditions[i].condition_str, tree=child)
            elif child.value is not None:
                self.x_off = self.x_off + 1 / self.tree_width
                self.plot_node(self.label_dict[child.value], (self.x_off, self.y_off), center_point, leaf_node)
                self.plot_mid_text((self.x_off, self.y_off), center_point, tree.branch_conditions[i].condition_str)
        self.y_off = self.y_off + 1 / self.tree_depth

    def plot_node(self, node_txt, center_point, parent_point, node_type):
        self.ax1.annotate(
            node_txt,
            xy=parent_point,
            xycoords='axes fraction',
            xytext=center_point,
            textcoords='axes fraction',
            va="center",
            ha="center",
            bbox=node_type,
            arrowprops=arrow_args
        )


    def plot_mid_text(self, center_point, parent_point, txt):
        x_mid = (parent_point[0] - center_point[0]) / 2 + center_point[0]
        y_mid = (parent_point[1] - center_point[1]) / 2 + center_point[1]
        self.ax1.text(x_mid, y_mid, txt)

    def create_plot(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)

        self.plot_tree((0.5, 1), "", tree=self.whole_tree)
        plt.show()
