import math
from matplotlib import pyplot as plt
from sklearn import tree
import numpy as np


class TreeNode:
    def __init__(self, feature_index, boundary, entropies, greaterLabel):
        self.feature_index = feature_index
        self.boundary = boundary
        self.entropies = entropies
        # label when goes to greater direction
        self.greaterLabel = greaterLabel
        self.level = -1
        # draw boundary to left/down or right/up?
        self.isOrientedAsc = False

    def set_level(self, level):
        # level of the node - starts from zero
        self.level = level

    def set_orientation(self, isOrientedAsc):
        # draw boundary to left/down or right/up?
        self.isOrientedAsc = isOrientedAsc

    def print(self):
        # print node information
        if self.feature_index == 0:
            feature = "X"
        else:
            feature = "Y"
        print("-----Level: {}, Feature Index: {}-----\nBoundary: {:.3f} "
              "EntropyLeft/Bottom: {:.3f}, EntropyRight/Top: {:.3f} EntropyTotal: {:.3f}"
              .format(self.level, self.feature_index, self.boundary, self.entropies[2], self.entropies[1], self.entropies[0]))
        print("{}>{:.3f} -> {}".format(feature, self.boundary, self.greaterLabel))


def read_data(data_path):
    data_file = open(data_path)
    dataset = []
    for line in data_file:
        lparts = line.strip().split(",")
        dataset.append([float(lparts[0]), float(lparts[1]), lparts[-1]])

    return dataset


def generate_dataset():
    # generate a dataset of two squares
    np.random.seed(10)
    outer_rect = np.random.uniform(low=5, high=11, size=(300, 2)).tolist()
    np.random.seed(10)
    inner_rect = np.random.uniform(low=7, high=9, size=(160, 2)).tolist()

    # remove the intersected region
    outer_rect = [[e[0], e[1], "animal"] for e in outer_rect
                  if e[0] < 7 or e[0] > 9 or e[1] < 7 or e[1] > 9]
    # re-format
    inner_rect = [[e[0], e[1], "panda"] for e in inner_rect]

    dataset = outer_rect + inner_rect

    return dataset


def calculate_part_entropy(dataset):
    counter = count_labels(dataset)
    entropy = 0.0
    for label in counter.keys():
        prob = counter[label] / len(dataset)
        entropy += -1 * prob * math.log(prob, 2)

    return entropy


def calculate_entropy(dataset, b_index, f_index):
    # include points with the same value
    boundary_value = 0 if b_index >= len(dataset) else dataset[b_index][f_index]
    while b_index < len(dataset) and dataset[b_index][f_index] == boundary_value:
        b_index += 1
    # entropy for each region
    entropy1 = calculate_part_entropy(dataset[:b_index])
    entropy2 = calculate_part_entropy(dataset[b_index:])
    # calculate weights
    w1 = b_index / len(dataset)
    w2 = (len(dataset) - b_index) / len(dataset)
    # weighted sum of entropies
    total_entropy = w1 * entropy1 + w2 * entropy2

    return total_entropy, entropy1, entropy2


def diversity_check(dataset):
    # check if the all labels are same
    label = dataset[0][-1]
    for point in dataset:
        if label != point[-1]:
            return False
    return True


def count_labels(dataset):
    # make a label-count dictionary
    counter = dict()
    for i in range(len(dataset)):
        if dataset[i][2] in counter.keys():
            counter[dataset[i][2]] += 1
        else:
            counter[dataset[i][2]] = 1
    return counter


def find_region_label(dataset):
    # vote for a label based on majority
    counter = count_labels(dataset)
    max_val = 0
    max_label = ""
    for label in counter.keys():
        if counter[label] > max_val:
            max_label = label
            max_val = counter[label]

    return max_label


def fit_boundary(subset, feature_index):
    # entropy_total, entropy_right, entropy_left
    prev_entropies = [float("inf"), float("inf"), float("inf")]
    b_index = 0
    # include points one by one and calculate entropy to determine the boundary
    for i in range(len(subset)):
        new_entropy_total, new_entropy_1, new_entropy_2 = calculate_entropy(subset, i, feature_index)
        if new_entropy_total <= prev_entropies[0]:
            prev_entropies = [new_entropy_total, new_entropy_1, new_entropy_2]
            b_index = i

    # maximize the boundary distance between the closest points to boundary
    prev_value = 0 if b_index == 0 else subset[b_index + 1][feature_index]
    boundary = (prev_value + subset[b_index][feature_index]) / 2

    greaterLabel = find_region_label(subset[:b_index])  # assign label to the region
    # create a node with the boundary
    node = TreeNode(feature_index=feature_index, boundary=boundary,
                    entropies=prev_entropies, greaterLabel=greaterLabel)

    return node, b_index


def select_best_subregion(subregion1, node1, subregion2, node2):
    if node1 is not None and node2 is not None:  # both regions have splits
        sub1_total_entropy = calculate_entropy(subregion1, len(subregion1), node1.feature_index)
        sub1_entropy_diff = sub1_total_entropy[0] - node1.entropies[0]
        sub2_total_entropy = calculate_entropy(subregion2, len(subregion2), node2.feature_index)
        sub2_entropy_diff = sub2_total_entropy[0] - node2.entropies[0]
        if sub1_entropy_diff > sub2_entropy_diff:
            node1.set_orientation(isOrientedAsc=True)  # the boundary goes to max limit
            return node1
        else:
            node2.set_orientation(isOrientedAsc=False)  # the boundary goes to min limit
            return node2
    elif node1 is None:  # only te subregion2 has a split
        node2.set_orientation(isOrientedAsc=False)  # the boundary goes to min limit
        return node2
    else:  # only te subregion1 has a split
        node1.set_orientation(isOrientedAsc=True)  # the boundary goes to max limit
        return node1


def construct_tree_with_feature_select(dataset):
    # check the first feature
    x_set = sorted(dataset, key=lambda triple: triple[0])
    x_set.reverse()
    node1x, index1x = fit_boundary(x_set, feature_index=0)
    node1x.set_orientation(isOrientedAsc=True)

    # check the second feature
    y_set = sorted(dataset, key=lambda triple: triple[1])
    y_set.reverse()
    node1y, index1y = fit_boundary(y_set, feature_index=1)
    node1y.set_orientation(isOrientedAsc=True)

    # find the one that lowers the entropy most
    if node1y.entropies[0] > node1x.entropies[0]:
        return node1x, index1x, list(x_set)
    else:
        return node1y, index1y, list(y_set)


def construct_decision_tree(dataset):

    tree_nodes = []

    # find zero level node
    node0, index0, datasetx = construct_tree_with_feature_select(dataset)
    node0.set_level(level=0)
    tree_nodes.append(node0)

    subregion1 = datasetx[:index0]  # left region
    subregion2 = datasetx[index0:]  # right region

    # check each subregion for each feature
    node1t1, node1t2 = None, None
    if not diversity_check(subregion1):  # if not all points have the same label
        node1t1, _, _ = construct_tree_with_feature_select(subregion1)
    if not diversity_check(subregion2):  # if not all points have the same label
        node1t2, _, _ = construct_tree_with_feature_select(subregion2)

    # determine the best split
    node1 = select_best_subregion(subregion1, node1t1, subregion2, node1t2)
    node1.set_level(level=1)
    tree_nodes.append(node1)

    return tree_nodes


def setup_matplotlib():
    # settings for plotting
    size = 6
    plt.rc("font", size=size)
    plt.rc("axes", titlesize=size)
    plt.rc("axes", labelsize=size)
    plt.rc("xtick", labelsize=size)
    plt.rc("ytick", labelsize=size)
    plt.rc("figure", titlesize=size)

    # one for dataset and one for boundaries
    subplt = plt.subplots(1, 2, squeeze=False)[1].flatten()

    return subplt


def draw_plot(axs, dataset, nodes, title, draw_boundaries=False):
    # prepare for scatter
    points = dict()
    for point in dataset:
        if point[2] in points.keys():
            points[point[2]].extend([point[0], point[1]])
        else:
            points[point[2]] = [point[0], point[1]]

    for label in points.keys():
        axs.scatter(points[label][::2], points[label][1::2], s=3, label=label)

    axs.margins(5e-3)  # this is to limit inner paddings
    axs.legend()
    axs.set_title(title+"\n", fontsize=10)
    axs.set(xlabel="X", ylabel="Y")

    if draw_boundaries:
        # first boundary is full length
        draw_bound(axs, nodes[0], axs.get_xlim(), axs.get_ylim(), 1, 3)
        for node_i in range(1, len(nodes)):
            tnode = nodes[node_i]
            pnode = nodes[node_i-1]
            # full length check
            if tnode.feature_index == pnode.feature_index:
                draw_bound(axs, tnode, axs.get_xlim(), axs.get_ylim(), node_i+1, 1)
                continue
            # find limits of the boundaries
            if nodes[1].isOrientedAsc:
                draw_bound(axs, tnode, (axs.get_xlim()[1], nodes[0].boundary), (axs.get_ylim()[1], nodes[0].boundary), node_i+1, 1)
            else:
                draw_bound(axs, tnode, (axs.get_xlim()[0], nodes[0].boundary), (axs.get_ylim()[0], nodes[0].boundary), node_i+1, 1)


def draw_bound(axs, cnode, draw_limits_x, draw_limits_y, bno, line_size):
    # draw either horizontal or vertical
    if cnode.feature_index == 0:
        axs.plot((cnode.boundary, cnode.boundary), (draw_limits_y), "k-", linewidth=line_size)
        axs.annotate("Boundary {}\nx={:.2f}".format(bno, cnode.boundary), xy=(cnode.boundary, axs.get_ylim()[0]),
                     ha="left", fontsize=8)
    else:
        axs.plot((draw_limits_x), (cnode.boundary, cnode.boundary), "k-", linewidth=line_size)
        axs.annotate("Boundary {}\ny={:.2f}".format(bno, cnode.boundary), xy=(axs.get_xlim()[1], cnode.boundary),
                     va="center", fontsize=8)


def compare_with_sklearn(dataset, max_depth=2):
    data = [[point[0], point[1]] for point in dataset]
    labels = [point[2] for point in dataset]

    # construct with scikit-learn
    dec_tree = tree.DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
    dec_tree = dec_tree.fit(data, labels)

    print("\nScikit-learn's Result")
    print(tree.export_text(dec_tree))


if __name__ == '__main__':

    __dataset = generate_dataset()
    # uncomment below to read data from file
    # __dataset = read_data("data.txt")

    # construct tree
    __nodes = construct_decision_tree(__dataset)
    print("My Implementation")
    for node_ in __nodes:
        node_.print()

    # draw plots
    sub_plots = setup_matplotlib()
    draw_plot(sub_plots[0], __dataset, None, "Dataset", draw_boundaries=False)
    draw_plot(sub_plots[1], __dataset, __nodes, "Boundaries", draw_boundaries=True)
    plt.show()

    # compare results
    compare_with_sklearn(__dataset, max_depth=2)
