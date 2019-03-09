"""Graph cells in clusters across days."""
import matplotlib.pyplot as plt
import numpy as np
import os.path as opath
import scipy.stats

from flow import misc
from flow import sorters
import flow.categories
import flow.misc.math
import flow.grapher as grapher
import flow.netcom as netcom
import flow.paths
import pool.config
import pool.database
import pool.plotting.colors as colors


def match_cluster_numbers(com1, cells1, com2, cells2, mxclust=100):
    """
    Match the cluster numbers across a pair of days.

    Parameters
    ----------
    com1 : array of ints
        Community labels for day 1
    cells1 : array of ints
        Cross-day cell labels for day 1
    com2 : array of ints
        Community labels for day 2
    cells2 : array of ints
        Cross-day cell labels for day 2
    mxclust : int
        The maximum size of a cluster (for sorting)

    Returns
    -------
    array of ints
        Correct cluster numbers for day 2

    """

    if np.sum(np.isfinite(com1)) == 0 or np.sum(np.isfinite(com2)) == 0:
        return com2

    match1 = com1[cells1]
    com2 += mxclust
    match2 = com2[cells2]
    matches1 = np.unique(match1[np.isfinite(match1)])
    matches2 = np.unique(match2[np.isfinite(match2)])

    # Move unmatched clusters from day 1 to the end for easier sorting
    # Then resort sequentially for easier bug-checking
    for c in np.unique(com1[np.isfinite(com1)]):
        if c not in matches1:
            com1[com1 == c] += mxclust

    for i, c in enumerate(np.unique(com1[np.isfinite(com1)])):
        com1[com1 == c] = i

    match1 = com1[cells1]
    matches1 = np.unique(match1[np.isfinite(match1)])

    # Renumber the clusters of day 2 to maximize the chance of matching
    done = []
    for m1 in matches1:
        mode2, count2 = scipy.stats.mode(match2[match1 == m1])

        if len(mode2) > 0:
            mxmode1, mxm2, mxcount1 = 0, 0, 0
            for m2 in matches2:
                mode1, count1 = scipy.stats.mode(match1[match2 == m2])

                if len(mode1) > 0 and count1[0] > mxcount1:
                    mxmode1, mxm2, mxcount1 = mode1[0], m2, count1[0]

            if count2[0] >= mxcount1:
                if m1 not in done:
                    match2[match2 == mode2] = m1
                    com2[com2 == mode2] = m1
                    done.append(m1)
            else:
                if mxmode1 not in done:
                    match2[match2 == m2] = mxmode1
                    com2[com2 == mode2] = mxmode1
                    done.append(mxmode1)

    # And make the remaining clusters sequential
    c2low = com2[com2 < mxclust]
    mx = max(np.nanmax(com1), np.nanmax(c2low) if len(c2low) > 0 else -1)
    fin2 = com2[np.isfinite(com2)]
    for i, c in enumerate(np.unique(fin2[fin2 >= mxclust])):
        com2[com2 == c] = i + mx

    return com2


def order_first_day(com1, cells1, com2, cells2, mxclust=100):
    """
    Order cells on the first day such that cells found across pairs of days are matched.

    Parameters
    ----------
    com1 : array of ints
        Community labels for day 1
    cells1 : array of ints
        Cross-day cell labels for day 1
    com2 : array of ints
        Community labels for day 2
    cells2 : array of ints
        Cross-day cell labels for day 2
    mxclust : int
        The maximum size of a cluster (for sorting)

    Returns
    -------
    tuple (array of ints, array of ints)
        The updated community labels for day 2 and
        cell numbers for ordering for all cells on day 1 to be included
    """

    com2 = match_cluster_numbers(com1, cells1, com2, cells2, mxclust)
    mx = np.max([len(com1) + np.nanmax(com1) + 1, len(com2) + np.nanmax(com2) + 1, mxclust*10])

    # Move the clusters into order
    ord1 = np.arange(len(com1), dtype=np.float64)
    for c in np.unique(com1[np.isfinite(com1)]):
        ord1[com1 == c] += c*mx

        # Within each cluster, move matched cells to top
        for cell1 in np.arange(len(com1))[com1 == c]:
            if cell1 not in cells1:
                ord1[cell1] += mxclust

    # Move cells in the non-matching clusters to the end
    match1 = com1[cells1]
    non_matches = np.setdiff1d(np.unique(com1), np.unique(match1))
    for i in non_matches:
        # Make sure to move to the end
        ord1[com1 == i] += mx*(len(np.unique(com1[np.isfinite(com1)])) + 2)

    # Limit to only the non-nan cells
    ord1 = np.argsort(ord1)
    out = np.arange(len(com1))[ord1]
    out = out[np.isfinite(com1[ord1])]

    return com2, out


def order_second_day(com1, cells1, com2, cells2, ord1):
    """
    Order the cells in the second day based on their pairings with the first.

    Parameters
    ----------
    com1 : array of ints
        Community labels for day 1
    cells1 : array of ints
        Cross-day cell labels for day 1
    com2 : array of ints
        Community labels for day 2
    cells2 : array of ints
        Cross-day cell labels for day 2
    ord1 : array of ints
        Order of cells on day 1

    Returns
    -------
    array of ints)
        Cell numbers for ordering for all cells on day 2 to be included
    """

    cells2_sorton_ord1 = []
    for i in ord1:
        if i in cells1:
            cells2_sorton_ord1.append(cells2[cells1 == i][0])

    pos = np.arange(len(com2))
    ord2 = []
    for c in np.unique(com2[np.isfinite(com2)]):
        ord2.extend([i for i in cells2_sorton_ord1 if i in pos[com2 == c]])
        ord2.extend([i for i in pos[com2 == c] if i not in cells2_sorton_ord1])

    return np.array(ord2)


def cluster_colors(andb, day, com, ord, color_by='cluster', categorize=('lick', 'ensure', 'reward', 'non')):
    """
    Get a list of colors for an ordered set of cells for a day.

    Parameters
    ----------
    andb : Analysis Database instance
    day : Day instance
    com : array of ints or floats
        The cluster each cell is in
    ord : array if ints or floats
        The order by which the cells should be sorted
    color_by : str
        Color cells by reactivation, cluster, or ratio
    categorize : tuple of strings
        The order into which cells should be categorized for coloring

    Returns
    -------
    list of strings
        The color code for each cell in ord

    """

    out = []
    if color_by == 'cluster':
        lbls = flow.categories.labels(day, categorize, cluster_numbers=com)

        for c in ord:
            out.append(pool.config.color(lbls[c]))
    elif color_by == 'repcount':
        count = andb.get('repcount_plus', day.mouse, day.date)

        for c in ord:
            out.append(colors.tuple2hex(colors.merge(
                pool.config.color('mint'), count[c]/10.0, mergeto=(0, 0, 0))))

    return out


class NodeDrawer:
    """
    Draw nodes on a graph
    """

    def __init__(self):
        """
        Create a graphing object.
        """

        self.fig = plt.figure(figsize=(11, 8))
        self.ax = plt.subplot(111)
        self.ntimes = 0
        self.ncells = 0

    def add(self, com1, cells1, com2, cells2, ord1, ord2, clrs1, clrs2, add_break=False):
        """
        Add a new day/pair of days.

        Parameters
        ----------
        com1 : array of ints
            Community labels for day 1
        cells1 : array of ints
            Cross-day cell labels for day 1
        com2 : array of ints
            Community labels for day 2
        cells2 : array of ints
            Cross-day cell labels for day 2
        ord1 : array of ints
            Order of cells on day 1
        ord2 : array of ints
            Order of cells on day 2
        clrs1 : list of colors
            Colors for each cell of day 1
        clrs2 : list of colors
            Colors for each cell of day 2
        add_break : bool
            If true, recalculate the sorting of clusters and plot 1 and 2

        """

        # Keep track of the number of cells
        if len(cells1) > self.ncells:
            self.ncells = len(cells1)
        if len(cells2) > self.ncells:
            self.ncells = len(cells2)

        # Plot day 1 if necessary
        if self.ntimes == 0 or add_break:
            self.ntimes += 1
            self._plot_day(com1, clrs1)

        # Plot day 2
        self.ntimes += 1
        self._plot_day(com2, clrs2)

        # Plot pairs
        for i, cell1 in enumerate(ord1):
            if cell1 in cells1:
                match2 = cells2[cells1 == cell1][0]
                if match2 in ord2:
                    pos = np.arange(len(ord2))[ord2 == match2]
                    self.ax.plot([2*(self.ntimes - 1) - 1, 2*self.ntimes - 1], [-i, -pos], color='#000000')

    def save(self, path):
        """
        Save the graph to a pdf file.

        Parameters
        ----------
        path : str
            Location to save file

        Returns
        -------

        """

        if path[-4:] != '.pdf':
            path += '.pdf'

        plt.savefig(path, transparent=True)

    def _plot_day(self, com, clrs):
        """
        Plot the cells and clusters of a day on an axis.

        Parameters
        ----------
        com : array of ints
            Community labels for day
        clrs : list of colors
            Colors for each cell of day
        """

        for i, clr in enumerate(clrs):
            self.ax.scatter([2*self.ntimes - 1], [-1*i], color=clr)

        run = 0
        coms = np.unique(com[np.isfinite(com)])
        for c in coms[:-1]:
            run += np.sum(com == c)
            self.ax.plot([2*self.ntimes - 1.5, 2*self.ntimes - 0.5], [-run + 0.5, -run + 0.5], color='#7c7c7c')


def main():
    """Main function."""
    args = parse_args()
    andb = pool.database.db()
    sorter = sorters.DatePairSorter.frommeta(
        mice=args.mice, dates=args.dates, day_distance=args.day_distance, sequential=args.sequential,
        cross_reversal=args.cross_reversal, tags=args.tags)
    np.warnings.filterwarnings('ignore')

    last_date, gr = None, NodeDrawer()
    for day1, day2 in sorter:
        if day1.date != last_date:
            nodes1 = andb.get('vdrive_%s'%args.stimulus, day1.mouse, day1.date) > args.visually_driven
            corrs1 = andb.get('noisecorr_%s'%args.stimulus, day1.mouse, day1.date)
            nodes1 = np.arange(len(nodes1))[nodes1]
            com1 = netcom.nxgraph(np.shape(corrs1)[0], corrs1, nodes1).communities()

            nodes2 = andb.get('vdrive_%s'%args.stimulus, day2.mouse, day2.date) > args.visually_driven
            corrs2 = andb.get('noisecorr_%s'%args.stimulus, day2.mouse, day2.date)
            nodes2 = np.arange(len(nodes2))[nodes2]
            com2 = netcom.nxgraph(np.shape(corrs2)[0], corrs2, nodes2).communities()

            com2, ord1 = order_first_day(com1, day1.cells, com2, day2.cells)
            clrs1 = cluster_colors(andb, day1, com1, ord1, args.color_by)
            add_break = True
        else:
            com1 = com2
            ord1 = ord2
            clrs1 = clrs2

            nodes2 = andb.get('vdrive_%s'%args.stimulus, day2.mouse, day2.date) > args.visually_driven
            corrs2 = andb.get('noisecorr_%s'%args.stimulus, day2.mouse, day2.date)
            nodes2 = np.arange(len(nodes2))[nodes2]
            com2 = netcom.nxgraph(np.shape(corrs2)[0], corrs2, nodes2).communities()

            com2 = match_cluster_numbers(com1, day1.cells, com2, day2.cells)
            add_break = False

        last_date = day2.date
        ord2 = order_second_day(com1, day1.cells, com2, day2.cells, ord1)
        clrs2 = cluster_colors(andb, day2, com2, ord2, args.color_by)
        gr.add(com1, day1.cells, com2, day2.cells, ord1, ord2, clrs1, clrs2, add_break)

    gr.save(opath.join(flow.paths.graphcrossday(), 'cluster-xday-%s-%i-%s' %
                       (day1.mouse, args.visually_driven, args.color_by)))


def parse_args():
    arg_parser = misc.default_parser(
        description="""
        Plot clusters across pairs of days.""",
        arguments=('mice', 'tags', 'dates'))
    arg_parser.add_argument(
        '-D', '--day_distance', nargs=2, type=int, default=(0, 6),
        help='Distance between days, inclusive.')
    arg_parser.add_argument(
        '-s', '--sequential', action="store_false",
        help='Limit only to sequential recording days.')
    arg_parser.add_argument(
        '-r', '--cross_reversal', action="store_true",
        help='Allow day pairs across reversal if true.')
    arg_parser.add_argument(
        '-cs', '--stimulus', default='plus',
        help='Stimulus used for clustering, i.e. plus, neutral, minus, or hofer (for cross-stimulus clustering)')
    arg_parser.add_argument(
        '-v', '--visually_driven', type=int, default=50,
        help='The visual-drivenness threshold. Standard is 50.')
    arg_parser.add_argument(
        '-c', '--color_by', type=str, default='cluster',
        help='Color cell names by "cluster", "repcount", or "ratio".')

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    main()
