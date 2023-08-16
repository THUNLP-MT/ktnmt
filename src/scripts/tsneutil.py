from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")


MOUSE_tedX_COLORS = {
    0: "#AEADD7",   # 紫色系
    1: "#908FD7",
    2: "#8C81D7",

    3: "#89B1DA",   # 蓝色系
    4: "#5796DA",
    5: "#1372CF",

    6: "#BEE89E",   # 绿色系
    7: "#6FC5AC",
    8: "#68BF60",

    9: "#FBE26F",   # 黄色系
    10: "#FDBA21",

    11: "#F6B3BD",  # 粉色系
    12: "#F689A8",
    13: "#F66B8A",
    14: "#000000"

}

MOUSE_15X_COLORS = {
    0: "#AEADD7",   # 紫色系
    1: "#908FD7",
    2: "#8C81D7",

    3: "#89B1DA",   # 蓝色系
    4: "#5796DA",
    5: "#1372CF",

    6: "#BEE89E",   # 绿色系
    7: "#6FC5AC",
    8: "#68BF60",

    9: "#FBE26F",   # 黄色系
    10: "#FDBA21",

    11: "#F6B3BD",  # 粉色系
    12: "#F689A8",
    13: "#F66B8A",
    14: "#000000"
    #14: "#ACACAC"   # 灰色

}

MOUSE_19X_COLORS = {
    0: "#AEADD7",   # 紫色系
    1: "#908FD7",
    2: "#8C81D7",

    3: "#89B1DA",   # 蓝色系
    4: "#5796DA",
    5: "#1372CF",

    6: "#BEE89E",   # 绿色系
    7: "#6FC5AC",
    8: "#68BF60",

    9: "#FBE26F",   # 黄色系
    10: "#FDBA21",
    11: "#FD9807",

    12: "#F6B3BD",  # 粉色系
    13: "#F689A8",
    14: "#F66B8A",

    15: "#CD493B",  # 砖红
    16: "#E24C3F",  # 珊瑚红
    17: "#747474",  # 深灰
    18: "#ACACAC",   # 灰色
    19: "#000000"
}

MOUSE_41X_COLORS = {
    0: "#AEADD7",   # 紫色系
    1: "#908FD7",
    2: "#8C81D7",
    3: "#6C62D7",
    4: "#6F34D7",

    5: "#BDD0F1",   # 蓝色系
    6: "#5B9DD1",
    7: "#357CBE",
    8: "#1C5BAA",
    9: "#0A447F",

    10: "#BEE89E",   # 绿色系
    11: "#6FC5AC",
    12: "#68BF60",
    13: "#20732C",
    14: "#07341D",


    15: "#FBE26F",   # 黄色系
    16: "#FDBA21",
    17: "#FD9807",
    18: "#BD3800",
    19: "#832501",

    20: "#F6B3BD",  # 粉色系
    21: "#F689A8",
    22: "#F66B8A",
    23: "#F14D92",

    24: "#CD493B",  # 砖红
    25: "#E24C3F",  # 珊瑚红

    26: "#C1C1C3",  # 灰
    27: "#ABABAB",
    28: "#747474",

    29: "#2E857C",  # 松绿
    30: "#0C524A",  # 暗绿


    31: "#EDDCBC",
    32: "#DBB97A",
    33: "#FFF294",
    34: "#A5DEBD",
    35: "#FDC296",
    36: "#D6BEE2",
    37: "#F0BBE1",
    38: "#A1BEE0",
    39: "#DAF4CF",
    40: "#F7FDA2",
    41: "#000000"
    #41: "#FFFFA7"
}


def plot(
        x,
        y,
        ax=None,
        title=None,
        draw_legend=True,
        draw_centers=False,
        draw_cluster_labels=False,
        colors=None,
        legend_kwargs=None,
        label_order=None,
        **kwargs
):
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, marker='^', rasterized=True, **plot_params)
    # markers = {}
    # for i in range(len(classes)):
    #     if i < int(len(classes)/2):
    #         markers[i] = '^'
    #     else:
    #         markers[i] = 'o'
    #
    # def mscatter(ax, x1, x2, m, c, **kw):
    #     import matplotlib.markers as mmarkers
    #     sc = ax.scatter(x1, x2, c=c, **kw)
    #     m = list(map(lambda x: m[x], y))
    #     if (m is not None) and (len(m) == len(y)):
    #         paths = []
    #         for marker in m:
    #             if isinstance(marker, mmarkers.MarkerStyle):
    #                 marker_obj = marker
    #             else:
    #                 marker_obj = mmarkers.MarkerStyle(marker)
    #             path = marker_obj.get_path().transformed(marker_obj.get_transform())
    #             paths.append(path)
    #         sc.set_paths(paths)
    #     return ax
    #
    # ax = mscatter(ax, x[:, 0], x[:, 1], c=point_colors, m=markers, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100]), ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100]), ax.axis("on")
    ax.set_xticklabels([]), ax.set_yticklabels([])

    lang = kwargs.get('lang')

    if draw_legend:
        legend_handles = [
            lines.Line2D(
                [],
                [],
                marker="^",
                markeredgewidth=0,
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=lang[int(yi)],
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

        plt.savefig('pic/{}.jpg'.format(kwargs.get("name")), dpi=300, format="jpg")
        plt.savefig('pic/{}.100dpi.eps'.format(kwargs.get("name")), dpi=100, format="eps")
        plt.savefig('pic/{}.300dpi.eps'.format(kwargs.get("name")), dpi=300, format="eps")
        plt.show()
