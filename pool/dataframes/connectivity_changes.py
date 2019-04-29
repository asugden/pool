import flow.paths as hardcodedpaths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import patsy

import flow.grapher
import flow.misc.math

from .. import config


class SubData:
    """
    Subset a dataframe by visual-drivenness, cluster, or reactivation.
    Save metadata about whether single-day, xday, or pairs were used.
    """

    def __init__(self, dataframe, xday=True, pairs=False, vdrive='plus'):
        """
        Initailize with the dataframe of all data and subset based on
        visual-drivenness defined in vdrive.

        Parameters
        ----------
        dataframe
        xday : bool
            True if cells are across days rather than for a single day
        pairs : bool
            True if entries are pairs of cells rather than single cells
        vdrive : str
            If not empty, limit to visually-driven cells of stimulus vdrive
        """

        self.orig_df = dataframe
        self.xday = xday
        self.pairs = pairs

        if len(vdrive) > 0:
            self.df = self.subset(vdrive)
        else:
            self.df = self.orig_df

    def subset(self, cs='', columns=None, positive=None, day_min=0, day_max=99,
               remove_lick=True, vdrive=50, multi_cluster=False, mouse=''):
        """
        Return a default dataframe subset and parameters.

        Parameters
        ----------
        cs : str
            Can be empty, 'plus', 'neutral', 'minus'
        columns : list
            Limit to a list of columns from which we drop NaNs
        positive : list
            List of columns that must be positive on both days
        day_min : int
            Minimum distance between days if xday, inclusive (>=)
        day_max : int
            Maximum distance between days if xday, inclusive (<=)
        remove_lick : bool
            If true, remove all licking cells
        vdrive : int
            Visual-drivenness threshold
        multi_cluster : bool
            If true, limit to only those days with both types of clusters

        Returns
        -------
        dataframe subset of original dataframe
        """

        # if self.xday:
        #     sub = self.orig_df.loc[(self.orig_df['day_distance'] >= day_min)
        #                            & (self.orig_df['day_distance'] <= day_max), :]
        # else:
        #     sub = self.orig_df.loc[:, :]

        sub = self.orig_df.loc[:, :]

        if len(mouse) > 0:
            sub = sub.loc[sub['mouse_day1'] == mouse, :]

        if len(cs) > 0:
            if self.xday:
                sub = sub.loc[(sub['visually_driven_%s_day1'%cs] >= vdrive)
                     & (sub['visually_driven_%s_day2'%cs] >= vdrive), :]
            else:
                sub = sub.loc[(sub['visually_driven_%s'%cs] >= vdrive), :]

        if remove_lick:
            if self.xday:
                sub = sub.loc[sub['base_lbl_day1'] != 'lick', :]
            else:
                sub = sub.loc[sub['base_lbl'] != 'lick', :]

        if multi_cluster:
            d1 = '_day1' if self.xday else ''

            for d in np.unique(sub['date%s'%d1].values):
                ddf = sub.loc[(sub['date%s'%d1] == d), :]

                if (len(ddf.loc[(ddf['base_lbl%s'%d1] == 'reward')
                            | (ddf['base_lbl%s'%d1] == 'ensure'), :]) == 0
                        or len(ddf.loc[ddf['base_lbl%s'%d1] == 'non', :]) == 0):

                    sub = sub.loc[sub['date%s'%d1] != d, :]

        if positive is not None:
            for lim in positive:
                if self.xday and 'd_' != lim[:2] and '_day1' not in lim and '_day2' not in lim:
                    sub = sub.loc[(sub['%s_day1'%lim] > 0) & (sub['%s_day2'%lim] > 0), :]
                else:
                    sub = sub.loc[sub[lim] > 0, :]

        if columns is not None:
            sub = sub.loc[:, columns].dropna()

        return sub

    def split(self, reactivation=None, cluster=None, within=None,
              combine_reward=True, reactivation_type='', combine_nons=False,
              reactivation_count=1, cs='plus'):
        """
        Split a dataframe by options reactviation or cluster.

        Parameters
        ----------
        reactivation : bool
            if True, split by reactivation, outer tuple if both cluster and reactivation
        cluster : bool
            if True, split by cluster, inner tuple if both cluster and reactivation
        within : bool
            Limit within clusters if true
        combine_reward : bool
            Combine ensure cells into reward cluster if splitting by cluster
        reactivation_type : str
            If empty, split on basic reactivation, can be 'reward-specific', 'reward-nonspecific', etc
            May include 'all'.
        reactivation_count : int
            Max number of reactivations to split on if reactivation_type is not all

        Returns
        -------
        List of [not reactivated, reactivated for each cluster]
            Clusters are in order of [non-reward, reward, [optional ensure if separated]]
        """

        d1, d2 = ('', '') if not self.xday else ('_day1', '_day2')
        sub_dfs = [self.df]
        out = []
        if cluster:
            if within and self.pairs:
                # Within group, pairs
                out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'non')
                                       & (self.df['base_lbl%s'%d2] == 'non'), :])

                if combine_reward:
                    out.append(self.df.loc[((self.df['base_lbl%s'%d1] == 'reward')
                                            | (self.df['base_lbl%s'%d1] == 'ensure'))
                                           & ((self.df['base_lbl%s'%d2] == 'reward')
                                            | (self.df['base_lbl%s'%d2] == 'ensure')), :])
                else:
                    out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'reward')
                                           & (self.df['base_lbl%s'%d2] == 'reward'), :])
                    out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'ensure')
                                           & (self.df['base_lbl%s'%d2] == 'ensure'), :])
            else:
                # Across groups if pairs, otherwise per-group
                out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'non'), :])

                if combine_reward:
                    out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'reward')
                                           | (self.df['base_lbl%s'%d1] == 'ensure'), :])
                else:
                    out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'reward'), :])
                    out.append(self.df.loc[(self.df['base_lbl%s'%d1] == 'ensure'), :])

        if reactivation:
            if len(out) > 0:
                sub_dfs = out
                out = []

            if 'all' in reactivation_type:
                for sub in sub_dfs:
                    # if len(sub.loc[sub['base_lbl%s'%d1] == 'reward', :]) > 0:
                    #     split1 = self.repname('plus', 'reward-specific')
                    #     split2 = self.repname('plus', 'nonreward-specific')
                    # else:
                    #     split1 = self.repname('plus', 'nonreward-specific')
                    #     split2 = self.repname('plus', 'reward-specific')
                    #
                    # out.append(sub.loc[(sub[split2] > 0), :])
                    # out.append(sub.loc[(sub[split1] == 0) & (sub[split2] == 0), :])
                    # out.append(sub.loc[(sub[split1] > 0), :])

                    if len(sub.loc[sub['base_lbl%s'%d1] == 'reward', :]) > 0:
                        split1 = self.repname(cs, 'reward-specific')
                    else:
                        split1 = self.repname(cs, 'nonreward-specific')

                    split2 = self.repname(cs, '')

                    out.append(sub.loc[(sub[split1] == 0) & (sub[split2] == 0), :])
                    out.append(sub.loc[(sub[split1] == 0) & (sub[split2] > 0), :])
                    out.append(sub.loc[(sub[split1] > 0), :])
            else:
                splitvar = self.repname(cs, reactivation_type)
                for sub in sub_dfs:
                    out.append(sub.loc[sub[splitvar] == 0, :])

                    for i in range(reactivation_count):
                        if i+1 < reactivation_count:
                            out.append(sub.loc[sub[splitvar] == i + 1, :])
                        else:
                            out.append(sub.loc[sub[splitvar] >= i + 1, :])

        if len(out) == 0:
            out = [self.df]

        if reactivation and cluster and combine_nons:
            non = pd.concat([dfi for dfi in out[::2]])
            out = [non] + [dfi for dfi in out[1::2]]

        return out

    def colors(self, reactivation=None, cluster=None, within=None,
                     combine_reward=True, reactivation_type='', combine_nons=False):
        """
        Return colors matching a split.

        Parameters
        ----------
        reactivation : bool
            if True, split by reactivation, outer tuple if both cluster and reactivation
        cluster : bool
            if True, split by cluster, inner tuple if both cluster and reactivation
        within : bool
            Limit within clusters if true
        combine_reward : bool
            Combine ensure cells into reward cluster if splitting by cluster
        reactivation_type : str
            If empty, split on basic reactivation, can be 'reward-specific', 'reward-nonspecific', etc

        Returns
        -------
        List of [not reactivated, reactivated for each cluster]
            Clusters are in order of [non-reward, reward, [optional ensure if separated]]
        """

        out = []
        if cluster:
            # Within group, pairs
            out.append('orange')

            if combine_reward:
                out.append('purple')
            else:
                out.append('purple')
                out.append('indigo')

        if reactivation:
            non_colors = {
                'mint': 'gray',
                'orange': 'gray',
                'purple': 'black',
                'indigo': 'blue',
            }

            if len(out) > 0:
                temp_colors = out
                out = []
            else:
                temp_colors = ['mint']

            splitvar = self.repname('plus', reactivation_type)
            for clr in temp_colors:
                out.append(non_colors[clr])
                out.append(clr)

        if len(out) == 0:
            out = ['mint']

        if reactivation and cluster and combine_nons:
            out = [out[0]] + [ci for ci in out[1::2]]

        return out

    def repname(self, cs='plus', reactivation_type=None):
        """
        Name of reactivation given all of the parameters

        Parameters
        ----------
        cs : str
            Stimulus name, such as 'plus'
        reactivation_type : str
            If empty: basic reactivation, can be 'reward-specific', 'reward-nonspecific', etc

        Returns
        -------
        str

        """

        d1 = '' if not self.xday else '_day1'

        if self.pairs:
            x = 'reppair_%s%s' % (cs, d1)
        else:
            x = 'repcount_%s%s' % (cs, d1)

        # Adjust x/replay value if necessary
        if reactivation_type is not None and len(reactivation_type) > 0:
            p = '_pair' if self.pairs else ''

            if reactivation_type == 'reward-specific':
                x = 'reward_specific_replay%s_%s%s' % (p, cs, d1)
            elif reactivation_type == 'reward-nonspecific':
                x = 'reward_nonspecific_replay%s_%s%s' % (p, cs, d1)
            elif reactivation_type == 'nonreward-specific':
                x = 'nonreward_specific_replay%s_%s%s' % (p, cs, d1)
            elif reactivation_type == 'nonreward-nonspecific':
                x = 'nonreward_nonspecific_replay%s_%s%s' % (p, cs, d1)
            else:
                x = reactivation_type

        return x

    def perday(self, x='hmm_dprime', y='stimulus_dff_2_4_plus', save='', split_reactivation=False,
               split_cluster=False, within=None, combine_reward=True, reactivation_type='',
               combine_nons=False, xrange=None, yrange=None):
        """
        Graph a plot per day of an x and y value

        Returns
        -------

        """

        subs = self.split(split_reactivation, split_cluster, within,
                          combine_reward, reactivation_type, combine_nons)
        clrs = self.colors(split_reactivation, split_cluster, within,
                           combine_reward, reactivation_type, combine_nons)

        d1 = '' if not self.xday else '_day1'
        gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
        for sub, clr in zip(subs, clrs):
            dates = sub.groupby('date%s' % d1).mean()
            xy = dates.loc[:, [x, y]].dropna().values
            gr.add(xy[:, 0], xy[:, 1], **{'color': clr})
            rp = flow.misc.math.nanspearman(xy[:, 0], xy[:, 1])
            print('Spearman correlation of %s:  %.3f p2=%.3g' % (clr, rp[0], rp[1]))

        if xrange is None:
            xrange = (None, None)
        if yrange is None:
            yrange = (None, None)

        gr.scatter(**{
            'xmin': xrange[0],
            'xmax': xrange[1],
            'save': save,
            'xtitle': x.replace('_', ' ').upper(),
            'ytitle': y.replace('_', ' ').upper(),
            'ymin': yrange[0],
            'ymax': yrange[1],
            # 'best-fit': True,
            'best-fits': True,
            'fit-type': 'glm'
        })

    def kde(self, y='stimulus_dff_2_4_plus', save='', split_reactivation=False,
            split_cluster=False, within=None, combine_reward=True, reactivation_type='',
            combine_nons=False, xrange=None, yrange=None):

        subs = self.split(split_reactivation, split_cluster, within,
                          combine_reward, reactivation_type, combine_nons)
        clrs = self.colors(split_reactivation, split_cluster, within,
                           combine_reward, reactivation_type, combine_nons)

        if xrange is None:
            xrange = (None, None)
        if yrange is None:
            yrange = (None, None)

        plot_kdes(subs, y, xrange, yrange, clrs, save)

    def bar(self, y='stimulus_dff_2_4_plus', save='', split_reactivation=False,
             split_cluster=False, within=None, combine_reward=True, reactivation_type='',
             combine_nons=False, yrange=None):

        subs = self.split(split_reactivation, split_cluster, within,
                          combine_reward, reactivation_type, combine_nons)
        clrs = self.colors(split_reactivation, split_cluster, within,
                           combine_reward, reactivation_type, combine_nons)

        pos = 1
        comb = []
        gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
        for sub, clr in zip(subs, clrs):
            vals = sub[y].dropna().values
            gr.add([pos], [np.nanmean(vals)], **{
                'color': clr, 'errors': [np.std(vals)/np.sqrt(len(vals))],
            })

            pos += 1
            comb.append(vals)

        comb = np.concatenate(comb).flatten()
        print np.shape(comb)
        gr.add([pos], [np.nanmean(comb)], **{'color': 'green', 'errors': [np.std(comb)/np.sqrt(len(comb))]})

        if yrange is None:
            yrange = (None, None)

        gr.bar(**{
            'save': save,
            'ytitle': y.replace('_', ' ').upper(),
            'ymin': yrange[0],
            'ymax': yrange[1],
        })

    def scatter(self, x='stim_decon_reward_plus_day1', y='stim_decon_reward_plus_day2',
                save='', split_reactivation=False, split_cluster=False, within=None,
                combine_reward=True, reactivation_type='', combine_nons=False, xrange=None, yrange=None):

        subs = self.split(split_reactivation, split_cluster, within,
                          combine_reward, reactivation_type, combine_nons)
        clrs = self.colors(split_reactivation, split_cluster, within,
                           combine_reward, reactivation_type, combine_nons)

        gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
        for sub, clr in zip(subs, clrs):
            xy = sub[[x, y]].dropna().values
            gr.add(xy[:, 0], xy[:, 1], **{'color': clr, 'opacity': 0.3})
            gr.add([np.nanmean(xy[:, 0])], [np.nanmean(xy[:, 1])], **{'color': clr, 'opacity': 1})

        if xrange is None:
            xrange = (None, None)
        if yrange is None:
            yrange = (None, None)

        gr.scatter(**{
            'save': save,
            'xtitle': x.replace('_', ' ').upper(),
            'ytitle': y.replace('_', ' ').upper(),
            'xmin': xrange[0],
            'xmax': xrange[1],
            'ymin': yrange[0],
            'ymax': yrange[1],
        })

    def ranksum(self, y='stim_decon_reward_plus_day2',
                split_reactivation=False, split_cluster=False, within=None,
                combine_reward=True, reactivation_type='', combine_nons=False, save=''):

        subs = self.split(split_reactivation, split_cluster, within,
                          combine_reward, reactivation_type, combine_nons)
        clrs = self.colors(split_reactivation, split_cluster, within,
                           combine_reward, reactivation_type, combine_nons)

        comb = []
        for i in range(len(subs)):
            y1 = subs[i][y].dropna().values
            comb.append(y1)
            print(i, clrs[i], len(y1), np.nanmean(y1), scipy.stats.wilcoxon(y1).pvalue/2)

            for j in range(i+1, len(subs)):
                y2 = subs[j][y].dropna().values
                print(i, j, clrs[i], clrs[j], scipy.stats.ranksums(y1, y2).pvalue/2)

        comb = np.concatenate(comb)
        print('all', len(comb), np.nanmean(comb), scipy.stats.wilcoxon(comb).pvalue/2)


def plot_kdes(dfs, y, xrange=(-0.5, 0.5), yrange=(0, 7),
                       colors=('orange', 'purple', 'gray', 'black'), save='', names=None, x=None):
    """
    Plot multiple 1-d KDE distributions on a single axis and save the result.

    Parameters
    ----------
    dfs : pandas dataframes
        Plot the y-values from each dataframe as a distribution
    y : str
        The name in the database of the y-value
    title : title of file to be saved
    xrange : tuple
        xmin, xmax
    yrange : tuple
        ymin, ymax
    colors : list of strings
        A list of colors to be plotted for each distribution
    names : list of strings
        Will be used just for describing statistical tests, optional
    x : an x-value, that will be used for determining NaNs if not None

    """

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')

    comp = []
    for i, df in enumerate(dfs):
        if x is not None:
            vals = df.loc[:, [x, y]].dropna()[y].values
        else:
            vals = df[y].dropna().values

        if len(vals) > 0:
            gr.add(vals, vals, **{'color': colors[i%len(colors)]})
            comp.append(vals)
        else:
            comp.append([])

    names = names if names is not None else colors
    for i in range(len(comp)):
        for j in range(i+1, len(comp)):
            if len(comp[i]) > 0 and len(comp[j]) > 0:
                print('Comparing %i/%s with %i/%s\nN: %i %i\tMu: %.4f %.4f\tStd: %.5f %.5f\tRS2: %.3g\tLevene2: %.3g' % (
                    i, names[i%len(names)], j, names[j%len(names)], len(comp[i]), len(comp[j]),
                    np.nanmean(comp[i]), np.nanmean(comp[j]), np.nanstd(comp[i]), np.nanstd(comp[j]), scipy.stats.ranksums(comp[i], comp[j]).pvalue,
                    scipy.stats.levene(comp[i], comp[j]).pvalue
                ))

    gr.histogram(**{'nbins': 50,
                    'xmin': xrange[0] if xrange[0] is not None else -0.5,
                    'xmax': xrange[1] if xrange[1] is not None else 0.5,
                    'ymin': yrange[0] if yrange[0] is not None else 0,
                    'ymax': yrange[1] if yrange[1] is not None else 5,
                    'smooth': True,
                    'cdf': True,
                    'ytitle': 'FREQUENCY',
                    'xtitle': y.upper().replace('_', ' '),
                    'save': save,
    })


def plot_kde_difference(dfs, y, xrange=(-0.5, 0.5), yrange=(0, 7),
                        colors=('orange', 'purple', 'gray', 'black'), save='', scale=1.0, below_zero=False):
    """
    Plot the difference ofKDE distributions relative to the first dataframe
    on a single axis and save the result.

    Parameters
    ----------
    dfs : pandas dataframes
        Plot the y-values from each dataframe as a distribution
    y : str
        The name in the database of the y-value
    xrange : tuple
        xmin, xmax
    yrange : tuple
        ymin, ymax
    colors : list of strings
        A list of colors to be plotted for each distribution
    save : str
        Title of graph to be saved
    scale : float
        Scale value to multiply all other values with

    """

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    x = np.linspace(xrange[0], xrange[1], 400)

    base_vals = dfs[0][y].dropna().values
    kde = sm.nonparametric.KDEUnivariate(base_vals)
    kde.fit(gridsize=min(len(base_vals), 1000))  # Estimate the densities
    nkx, nky = kde.support, kde.density
    fn = scipy.interpolate.interp1d(nkx, nky, bounds_error=False, fill_value=0)
    nky = fn(x)

    for df, clr in zip(dfs[1:], colors[1:]):
        vals = df[y].dropna().values
        kde = sm.nonparametric.KDEUnivariate(vals)
        kde.fit(gridsize=min(len(vals), 1000))  # Estimate the densities
        rkx, rky = kde.support, kde.density*scale

        fn = scipy.interpolate.interp1d(rkx, rky, bounds_error=False, fill_value=0)
        rky = fn(x)

        if not below_zero:
            diff = np.clip(rky - nky, 0, None)
        else:
            diff = rky - nky

        gr.add(x, diff, **{'color': clr})

    gr.line(**{
        'xmin': xrange[0],
        'xmax': xrange[1],
        'ymin': yrange[0],
        'ymax': yrange[1],
        'smooth': True,
        'ytitle': 'FREQUENCY',
        'xtitle': y.replace('_', ' ').upper(),
        'save': save,
    })


def levenedist(df, cs='plus', corr=None, gtzero=False, day_max=6, day_min=0, rep=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Plot a distribution of reactivated and non-reactivated cells/pairs.

    Parameters
    ----------
    df : dataframe
    cs : the stimulus to plot
    corr : correlation type
        Will be graph-clustering if it exists, otherwise the default is 'noise', or can be 'spont', 'noise-nolick'
    gtzero : bool
        If true, force y-values to be greater than 0 on both days
    day_max : maximum distance between days
    day_min : minimum distance between days

    """

    corr, x, y = _names(df, corr, rep, cs)
    sub = df.loc[(df['visually_driven_%s_day1'%cs] >= 50)
                 & (df['visually_driven_%s_day2'%cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if gtzero:
        sub = sub.loc[(sub['%s_day1' % y] > 0) & (sub['%s_day2' % y] > 0), :]

    non = sub.loc[(sub[x] == 0), 'd_%s'%y].dropna().values
    rep = sub.loc[(sub[x] > 0), 'd_%s'%y].dropna().values

    print('Levene distance of %s==0 (%i at %.4f) and %s>0 (%i at %.4f): %.3g' % (
        x, len(non), np.mean(non), x, len(rep), np.mean(rep), scipy.stats.levene(non, rep).pvalue))
    rs = scipy.stats.ranksums(non, rep)
    print('RS: %.3f, %.3f, %.3f p=%.3g' % (np.nanmean(non), np.nanmean(rep), rs[0], rs[1]/2.0))

    kde = sm.nonparametric.KDEUnivariate(non)
    kde.fit(gridsize=min(len(non), 1000))  # Estimate the densities
    nkx, nky = kde.support, kde.density

    kde = sm.nonparametric.KDEUnivariate(rep)
    kde.fit(gridsize=min(len(rep), 1000))  # Estimate the densities
    rkx, rky = kde.support, kde.density

    fn = scipy.interpolate.interp1d(nkx, nky, bounds_error=False, fill_value=0)
    nky = fn(rkx)
    diff = np.clip(rky - nky, 0, None)
    enh = np.sum(diff*(rkx[1] - rkx[0]))

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    gr.add(non, non, **{'color':'gray'})
    gr.add(rep, rep, **{'color':config.colors(cs), 'opacity':0.5})
    gr.histogram(**{'nbins':50,
                    'xmin': xmin if xmin is not None else -0.5 if corr != 'spont' else -0.05,
                    'xmax': xmax if xmax is not None else 0.5 if corr != 'spont' else 0.05,
                    'ymin':0,
                    'ymax': ymax if ymax is not None else 7 if corr != 'spont' else 60,
                    'smooth':True,
                    'cdf': True,
                    'ytitle':'FREQUENCY',
                    'xtitle':'CLUSTERING',
                    'save': '%s-vdrive-%s levene 2-t p=%.3g n non=%i, n rep=%i, std %.3f %.3f, enh= %.3f' %
                            (corr, cs, scipy.stats.levene(non, rep).pvalue,
                             len(non), len(rep),
                             np.std(non), np.std(rep), enh)})

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    gr.add(rkx, diff, **{'color':'gray'})
    gr.line(**{
        'xmin':-0.5 if corr != 'spont' else -0.05,
        'xmax':0.5 if corr != 'spont' else 0.05,
        'ymin':0,
        'ymax':7 if corr != 'spont' else 60,
        'smooth':True,
        'ytitle':'FREQUENCY',
        'xtitle':'CLUSTERING',
        'save': 'clust-vdrive-diff-%s'%(cs)
    })


def levenespecific(df, cs='plus', corr=None, gtzero=False, day_max=6, day_min=0):
    """
    Plot a distribution of reactivated and non-reactivated cells/pairs.

    Parameters
    ----------
    df : dataframe
    cs : the stimulus to plot
    corr : correlation type
        Will be graph-clustering if it exists, otherwise the default is 'noise', or can be 'spont', 'noise-nolick'
    gtzero : bool
        If true, force y-values to be greater than 0 on both days
    day_max : maximum distance between days
    day_min : minimum distance between days

    """

    corr, x, y = _names(df, corr, None, cs)
    pair = '_pair' if corr != 'clust' else ''
    sub = df.loc[(df['visually_driven_%s_day1'%cs] >= 50)
                 & (df['visually_driven_%s_day2'%cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if gtzero:
        sub = sub.loc[(sub['%s_day1' % y] > 0) & (sub['%s_day2' % y] > 0), :]

    non = sub.loc[(sub[x] == 0), 'd_%s'%y].dropna().values
    neg = sub.loc[(sub['nonreward_specific_replay%s_plus_day1'%pair] > 0), 'd_%s'%y].dropna().values
    pos = sub.loc[(sub['reward_specific_replay%s_plus_day1'%pair] > 0), 'd_%s'%y].dropna().values
    oth = sub.loc[(sub[x] > 0)
                  & (sub['reward_specific_replay%s_plus_day1'%pair] == 0)
                  & (sub['nonreward_specific_replay%s_plus_day1'%pair] == 0), 'd_%s'%y].dropna().values

    kde = sm.nonparametric.KDEUnivariate(non)
    kde.fit(gridsize=min(len(non), 1000))  # Estimate the densities
    okx, oky = kde.support, kde.density

    kde = sm.nonparametric.KDEUnivariate(neg)
    kde.fit(gridsize=min(len(neg), 1000))  # Estimate the densities
    nkx, nky = kde.support, kde.density

    kde = sm.nonparametric.KDEUnivariate(pos)
    kde.fit(gridsize=min(len(pos), 1000))  # Estimate the densities
    pkx, pky = kde.support, kde.density

    x = np.linspace(-0.5, 0.5, 400) if 'spont' not in corr else np.linspace(-0.02, 0.02, 400)

    fn = scipy.interpolate.interp1d(okx, oky, bounds_error=False, fill_value=0)
    oky = fn(x)
    fn = scipy.interpolate.interp1d(nkx, nky, bounds_error=False, fill_value=0)
    nky = fn(x)
    fn = scipy.interpolate.interp1d(pkx, pky, bounds_error=False, fill_value=0)
    pky = fn(x)

    diffn = np.clip(nky - oky, 0, None)
    diffp = np.clip(pky - oky, 0, None)

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    gr.add(non, non, **{'color':'gray'})
    gr.add(oth, oth, **{'color':'black', 'opacity':0.6})
    gr.add(neg, neg, **{'color':'orange', 'opacity':0.6})
    gr.add(pos, pos, **{'color':'purple', 'opacity':0.6})
    rsnp = scipy.stats.ranksums(neg, pos)
    rsno = scipy.stats.ranksums(neg, non)
    rspo = scipy.stats.ranksums(pos, non)
    print('Neg mu: %.3f\tNon mu: %.3f\tPos mu: %.3f\tRS neg-pos p=%.3g\tRS neg-non p=%.3g\tRS non-pos p=%.3g' %
          (np.nanmean(neg), np.nanmean(non), np.nanmean(pos), rsnp[1]/2.0, rsno[1]/2.0, rspo[1]/2.0))

    gr.histogram(**{'nbins':50,
                    'xmin':x[0],
                    'xmax':x[-1],
                    'ymin':0,
                    'ymax':5 if 'spont' not in corr else 60,
                    'smooth':True,
                    'cdf': True,
                    'ytitle':'FREQUENCY',
                    'xtitle':'CLUSTERING',
                    'save': '%s-vdrive-specifc-rep N-%i-%.3f O-%i-%.3f P-%i-%.3f NPp-%.3g NOp=%.3g NPp=%.3g' %
                            (corr, len(neg), np.nanmean(neg), len(non), np.nanmean(non), len(pos),
                             np.nanmean(pos), rsnp[1]/2.0, rsno[1]/2.0, rspo[1]/2.0)
    })
    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    gr.add(x, diffn, **{'color':'orange'})
    gr.add(x, diffp, **{'color': 'purple'})
    gr.line(**{
        'xmin':x[0],
        'xmax':x[-1],
        'ymin':0,
        'ymax':5 if 'spont' not in corr else 60,
        'smooth':True, 'ytitle':'FREQUENCY', 'xtitle':'CLUSTERING', 'save': '%s-vdrive-specific-diff' % corr})


def corr_cluster(df, corr=None, gtzero=False, clusternolick=False, day_max=6, day_min=0, rep=None, repcs=None):
    """
    Return the spearman correlation of clusters.

    Parameters
    ----------
    df : dataframe
    corr : correlation type
        Will be graph-clustering if it exists, otherwise the default is 'noise', or can be 'spont', 'noise-nolick'
    gtzero : bool
        If true, force y-values to be greater than 0 on both days
    clusternolick : bool
        If true, cluster using nolick noise correlations
    day_max : maximum distance between days
    day_min : minimum distance between days

    """

    cs = 'plus'
    corr, x, y = _names(df, corr, rep, cs)
    if repcs is not None:
        _, x, _ = _names(df, corr, rep, repcs)
    lbl = 'nolick_lbl' if clusternolick else 'base_lbl'

    sub = df.loc[(df['visually_driven_%s_day1' % cs] >= 50)
                 & (df['visually_driven_%s_day2' % cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if gtzero:
        sub = sub.loc[(sub['%s_day1' % y] > 0) & (sub['%s_day2' % y] > 0), :]

    if corr == 'clust':
        sub = sub[['%s_day1'%lbl, x, 'd_%s'%y]].dropna()
    else:
        sub = sub[['%s_day1'%lbl, '%s_cell2_day1'%lbl, x, 'd_%s'%y]].dropna()

    print('Across groups')
    # Each x-group
    for cluster in ['ensure', 'reward', 'non']:
        pos = sub['%s_day1'%lbl] == cluster
        xval = sub.loc[pos, x].values
        yval = sub.loc[pos, 'd_%s'%y].values
        c, p = flow.misc.math.nanspearman(xval, yval)
        print('%s group (n: %i, >0 %i)\tSpearman: %.3f, p=%.3g' % (cluster, len(xval),
                                                                   len(xval[xval > 0]), c, p/2.0))

    print('\tTwo-group version')
    pos = (sub['%s_day1'%lbl] == 'ensure') | (sub['%s_day1'%lbl] == 'reward')
    xval = sub.loc[pos, x].values
    yval = sub.loc[pos, 'd_%s'%y].values
    c, p = flow.misc.math.nanspearman(xval, yval)
    print('ensure/reward group (n: %i >0 %i)\tSpearman: %.3f, p=%.3g' % (len(xval),
                                                                         len(xval[xval > 0]), c, p/2.0))

    if corr != 'clust':
        print('\tWithin group')
        for cluster in ['ensure', 'reward', 'non']:
            pos = (sub['%s_day1'%lbl] == cluster) & (sub['%s_cell2_day1'%lbl] == cluster)
            xval = sub.loc[pos, x].values
            yval = sub.loc[pos, 'd_%s'%y].values
            c, p = flow.misc.math.nanspearman(xval, yval)
            print('%s group (n: %i)\tSpearman: %.3f, p=%.3g' % (cluster, len(xval), c, p/2.0))

        print('\tTwo-group version')
        pos = (((sub['%s_day1'%lbl] == 'ensure')
              | (sub['%s_day1'%lbl] == 'reward'))
              & ((sub['%s_cell2_day1'%lbl] == 'ensure')
              | (sub['%s_cell2_day1'%lbl] == 'reward')))
        xval = sub.loc[pos, x].values
        yval = sub.loc[pos, 'd_%s'%y].values
        c, p = flow.misc.math.nanspearman(xval, yval)
        print('ensure/reward group (n: %i)\tSpearman: %.3f, p=%.3g' % (len(xval), c, p/2.0))


def multikde(df, corr=None, rep=None, gtzero=False, combensure=True, day_max=7, day_min=0, within=False, repcs=None):

    cs = 'plus'
    corr, x, y = _names(df, corr, rep, cs)
    if repcs is not None:
        _, x, _ = _names(df, corr, rep, repcs)

    lbl = 'nolick_lbl' if False else 'base_lbl'
    xmin, xmax, ymin, ymax = 0, 20, -0.5, 0.5
    if 'spont' in corr:
        xmax, ymin, ymax = 5, -0.02, 0.02

    sub = df.loc[(df['visually_driven_%s_day1'%cs] >= 50)
                 & (df['visually_driven_%s_day2'%cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if gtzero:
        sub = sub.loc[(sub['%s_day1' % y] > 0) & (sub['%s_day2' % y] > 0), :]

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    clrs = {
        'reward': ['#266B54', 'green'],
        'non': ['#6B3105', 'orange'],
        'ensure': ['indigo', 'purple'],
    }

    for cluster in ['reward', 'non', 'ensure']:
        if not combensure or cluster != 'ensure':
            if combensure and cluster == 'reward':
                lims = ((sub['%s_day1'%lbl] == 'reward') | (sub['%s_day1'%lbl] == 'ensure'))
                if within:
                    lims = lims & ((sub['%s_day2'%lbl] == 'reward') | (sub['%s_day2'%lbl] == 'ensure'))

                clr1, clr2 = clrs['ensure']
            else:
                lims = (sub['%s_day1'%lbl] == cluster)
                if within:
                    lims = lims & (sub['%s_day2'%lbl] == cluster)
                clr1, clr2 = clrs[cluster]
            clust = sub.loc[lims, [x, 'd_%s'%y]].dropna()

            if 'nonspecific' in x:
                clr1, clr2 = '#666666', '#000000'

            kernel = scipy.stats.gaussian_kde(np.vstack([clust[x].values, clust['d_%s'%y].values]))
            Z = np.reshape(kernel(positions).T, X.shape)

            gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
            ax = gr.axis()
            # mesh = ax.pcolormesh(np.linspace(xmin, xmax, 100),
            #                  np.linspace(ymin, ymax, 100),
            #                  np.flipud(np.rot90(Z)),
            #                  cmap=colors.gradclr(clr1, clr2))
            ax.contour(np.linspace(xmin, xmax, 100),
                             np.linspace(ymin, ymax, 100),
                             np.flipud(np.rot90(Z)), 16, cmap=colors.gradclr(clr1, clr2))
            # gr.colorbar(mesh)
            gr.graph(ax, **{'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                            'save': '%s-%s-rep-%s-multicdf' % (corr, cluster, x)})




def levene_summary(df, cs='plus', corr=None, gtzero=False, day_max=6, day_min=0, rep=None):
    """
    Describe a distribution of reactivated and non-reactivated cells/pairs.

    Parameters
    ----------
    df : dataframe
    cs : the stimulus to plot
    corr : correlation type
        Will be graph-clustering if it exists, otherwise the default is 'noise', or can be 'spont', 'noise-nolick'
    gtzero : bool
        If true, force y-values to be greater than 0 on both days
    day_max : maximum distance between days
    day_min : minimum distance between days

    """

    corr, x, y = _names(df, corr, rep, cs)
    sub = df.loc[(df['visually_driven_%s_day1'%cs] >= 50)
                 & (df['visually_driven_%s_day2'%cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if gtzero:
        sub = sub.loc[(sub['%s_day1' % y] > 0) & (sub['%s_day2' % y] > 0), :]

    non = sub.loc[(sub[x] == 0), 'd_%s'%y].dropna().values
    rep = sub.loc[(sub[x] > 0), 'd_%s'%y].dropna().values

    print('Levene distance of non-reward (%i at %.4f) and reward clusters (%i at %.4f): %.3g' % (
        len(non), np.mean(non), len(rep), np.mean(rep), scipy.stats.levene(non, rep).pvalue))
    rs = scipy.stats.ranksums(non, rep)
    print('RS: %.3f, %.3f, %.3f p=%.3g\n' % (np.nanmean(non), np.nanmean(rep), rs[0], rs[1]/2.0))

    for mouse in sub.mouse_day1.unique():
        non = sub.loc[(sub[x] == 0) & (sub.mouse_day1 == mouse), 'd_%s'%y].dropna().values
        rep = sub.loc[(sub[x] > 0) & (sub.mouse_day1 == mouse), 'd_%s'%y].dropna().values

        print(mouse.upper())
        print('Levene distance of non-reward (%i at %.4f) and reward clusters (%i at %.4f): %.3g'%(
            len(non), np.mean(non), len(rep), np.mean(rep), scipy.stats.levene(non, rep).pvalue))
        rs = scipy.stats.ranksums(non, rep)
        print('RS: %.3f, %.3f, %.3f p=%.3g'%(np.nanmean(non), np.nanmean(rep), rs[0], rs[1]/2.0))


def replay_connectivity_plots(df, x=None, y=None, positive=None,
                              day_min=0, day_max=7, within=False, combine_reward=True):
    """
    Plot contour plots of replay by connection probability.

    Parameters
    ----------
    df : Pandas dataframe
    x : str
        Name of reactivation type, e.g. 'reward-specific'
    y : str
        Type of correlation, can be 'clust', 'noise', 'noise-nolick', or 'spont'
    positive : bool
        Force all values to be greater than 0 if True, automatically set to True for 'noise' in y
    day_min : int
        Minimum distance between days
    day_max : int
        Maximum distance between days
    within : bool
        If true, limit analyses within clusters
    combine_reward : bool
        If true, combine ensure cells into reward clusters.

    """

    sub, ytype, basex, y = _defaults(df, None, y, 'plus', positive, day_min, day_max)

    cluster_colors = ['orange', 'purple', 'indigo']
    clusters = _split(sub, ytype, basex, y, cluster=True, within=within,
                      combine_reward=combine_reward, values=False)

    vals, spec = [], ''
    if x is not None:
        pos = 0 if x[:3] == 'non' else 1
        spec = '-pos-specific' if pos == 1 else '-neg-specific'
        c_df = clusters[pos]

        cluster_colors = ['black', cluster_colors[pos]]
        vals = [c_df[[_reptype(ytype, x, specific=v), 'd_%s'%y]].dropna().values for v in [False, True]]
    else:
        for c_df in clusters:
            vals.append(c_df[[basex, 'd_%s'%y]].dropna().values)

    gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')
    ax = gr.axis()
    yran, xran = _ranges(ytype)
    X, Y = np.mgrid[xran[0]:xran[1]:100j, yran[0]:yran[1]:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    for clr, xy in zip(cluster_colors, vals):
        kernel = scipy.stats.gaussian_kde(np.vstack([xy[:, 0], xy[:, 1]]))
        Z = np.reshape(kernel(positions).T, X.shape)
        ax.contour(np.linspace(xran[0], xran[1], 100),
                   np.linspace(yran[0], yran[1], 100),
                   np.flipud(np.rot90(Z)), 16, colors=flow.grapher.color(clr), linewidths=2)

    gr.graph(ax, **{
        'xmin': xran[0],
        'xmax': xran[1],
        'ymin': yran[0],
        'ymax': yran[1],
        'save': '%s-connectivity-contour%s-%s' % (
            ytype, spec, 'across' if not within else 'within'),
    })


def cluster_distributions(df, x=None, y=None, positive=None,
                          day_min=0, day_max=7, within=False, combine_reward=True):
    """
    Plot KDE distributions across clusters, separated into two plots by reactivation.

    Parameters
    ----------
    df : Pandas dataframe
    x : str
        Refers to the replay type, can be 'reward-specifc'
    y : str
        Refers to the correlation type, can be 'noise', 'spont', 'noise-nolick', or 'clust'
    positive : bool
        Force all values to be greater than 0 if True, automatically set to True for 'noise' in y
    day_min : int
        Minimum distance between days
    day_max : int
        Maximum distance between days
    within : bool
        If true, limit analyses within clusters
    combine_reward : bool
        If true, combine ensure cells into reward clusters.

    """
    sub, ytype, x, y = _defaults(df, x, y, 'plus', positive, day_min, day_max)

    cluster_names = ['non', 'reward', 'ensure']
    cluster_colors = ['orange', 'purple', 'indigo']
    clusters = _split(sub, ytype, x, y, reactivation=True, cluster=True, within=within, combine_reward=combine_reward)

    for i, rep in enumerate(['norep', 'rep']):
        gr = flow.grapher.graph(hardcodedpaths.graphcrossday(), 'half')

        for j, (x, y) in enumerate(clusters[i::2]):
            gr.add(y, y, **{'color': cluster_colors[j]})
            if j > 0:
                print('Ranksums %s of %s (%i) and %s (%i): %.3g' % (
                    rep,
                    cluster_names[j-1], len(y),
                    cluster_names[j], len(y),
                    scipy.stats.ranksums(clusters[i::2][j-1][1], y).pvalue/2
                ))

        xran, yran = _ranges(ytype)
        gr.histogram(**{
            'nbins': 30,
            'xmin': xran[0],
            'xmax': xran[1],
            'ymin': yran[0],
            'ymax': yran[1],
            'smooth': True,
            'cdf': True,
            'ytitle': 'FREQUENCY',
            'xtitle': 'CLUSTERING',
            'save': '%s-cluster-cdf-%s-%s' % (ytype, rep, 'across' if not within else 'within')
        })


def subsample_to_match_ranksums(dfs, y, match_to=0.25, batch=20):
    """
    Subsample from dataframes after the first to match
    the distribution of value y in the first dataframe.

    Parameters
    ----------
    dfs : list of dataframes
        The first is the dataframe to be matched, all other dataframes will
        be subsampled to match the first.
    y : str
        The column to be matched
    match_to : float
        The 2-tailed rank sum p-value that must be exceeded
    batch : int
        The number of values to be removed before ranksumming again


    Returns
    -------
    list of dataframes
        A list of dataframes in which the first matches the first of dfs
        and the rest are subsampled (in the same order as dfs)
    """

    out = [dfs[0]]
    dist = dfs[0][y].dropna().values

    kde = sm.nonparametric.KDEUnivariate(dist)
    kde.fit(gridsize=min(len(dist), 1000))  # Estimate the densities
    fref = scipy.interpolate.interp1d(kde.support, kde.density,
                                      bounds_error=False, fill_value=0)

    for sub in dfs[1:]:
        sub = sub.dropna(subset=[y]).copy().reindex()
        x = np.linspace(min(np.nanmin(dist), np.nanmin(sub[y].dropna().values)),
                        max(np.nanmax(dist), np.nanmax(sub[y].dropna().values)),
                        100)
        ref_match = fref(x)

        while len(sub) > batch and scipy.stats.ranksums(dist, sub[y].values)[1] < match_to:
            print('Batch', len(sub))
            kde = sm.nonparametric.KDEUnivariate(sub[y].values)
            kde.fit(gridsize=min(len(sub), 1000))  # Estimate the densities
            fsub = scipy.interpolate.interp1d(kde.support, kde.density,
                                              bounds_error=False, fill_value=0)
            diff = fsub(x) - ref_match
            diff /= np.nanmax(diff)
            diff = np.clip(diff, 0, 1)
            fprob = scipy.interpolate.interp1d(x, diff, bounds_error=False, fill_value=0)

            dropped, counts = 0, 0
            while dropped < batch and counts < 10000:
                pos = np.random.randint(len(sub))
                counts += 1
                if np.random.random() < fprob(sub[y].iloc[pos]):
                    sub = sub.drop(sub.index[pos])
                    dropped += 1

        out.append(sub)
    return out


def _defaults(df, x=None, y=None, cs='plus', positive=None, day_min=0, day_max=99):
    """
    Return a default dataframe subset and parameters.

    Parameters
    ----------
    df : Pandas dataframe
    x : str
        Replay type such as 'reward-specific'
    y : str
        Can be 'clust', 'spont', 'noise', or 'noise-nolick', the latter of which require pairs
    cs : str
        Can be 'plus', 'neutral', 'minus'
    positive : bool
        If false, allow all noise correlations. If true, enforce noise correlations > 0.
        If None, sets to True if noise, False if spont or clust
    day_min : int
        Minimum distance between days, inclusive (>=)
    day_max : int
        Maximum distance between days, inclusive (<=)

    Returns
    -------
    dataframe, ytype, x name for dataframe, y name for dataframe
    """

    corr, x, y = _names(df, y, x, cs)
    sub = df.loc[(df['visually_driven_%s_day1'%cs] >= 50)
                 & (df['visually_driven_%s_day2'%cs] >= 50)
                 & (df['day_distance'] >= day_min)
                 & (df['day_distance'] <= day_max), :]

    if positive is None:
        positive = True if 'noise' in corr else False

    if positive:
        sub = sub.loc[(sub['%s_day1'%y] > 0) & (sub['%s_day2'%y] > 0), :]

    return sub, corr, x, y


def _split(df, ytype, x=None, y=None, reactivation=None, cluster=None, within=None,
           combine_reward=True, reactivation_type='', values=True):
    """
    Split a dataframe by options reactviation or cluster.

    Parameters
    ----------
    df : Pandas dataframe
    x : str
        From _defaults
    y : str
        From _defaults
    ytype : str
        From _defaults
    reactivation : bool
        if True, split by reactivation, outer tuple if both cluster and reactivation
    cluster : bool
        if True, split by cluster, inner tuple if both cluster and reactivation
    within : bool
        Limit within clusters if true
    combine_reward : bool
        Combine ensure cells into reward cluster if splitting by cluster
    reactivation_type : str
        If empty, split on basic reactivation, can be 'reward-specific', 'reward-nonspecific', etc.
    values : bool
        If true, convert to d_y values

    Returns
    -------
    List of [not reactivated, reactivated for each cluster]
        Clusters are in order of [non-reward, reward, [optional ensure if separated]]
    """

    sub_dfs = [df]
    out = []
    if cluster:
        if within and 'clust' not in ytype:
            out.append(df.loc[(df['base_lbl_day1'] == 'non') & (df['base_lbl_day2'] == 'non'), :])

            if combine_reward:
                out.append(df.loc[((df['base_lbl_day1'] == 'reward')
                                   | (df['base_lbl_day1'] == 'ensure'))
                                  & ((df['base_lbl_day2'] == 'reward')
                                     | (df['base_lbl_day2'] == 'ensure')), :])
            else:
                out.append(df.loc[(df['base_lbl_day1'] == 'reward') & (df['base_lbl_day2'] == 'reward'), :])
                out.append(df.loc[(df['base_lbl_day1'] == 'ensure') & (df['base_lbl_day2'] == 'ensure'), :])
        else:
            out.append(df.loc[(df['base_lbl_day1'] == 'non'), :])

            if combine_reward:
                out.append(df.loc[(df['base_lbl_day1'] == 'reward')
                                   | (df['base_lbl_day1'] == 'ensure'), :])
            else:
                out.append(df.loc[(df['base_lbl_day1'] == 'reward'), :])
                out.append(df.loc[(df['base_lbl_day1'] == 'ensure'), :])

    if reactivation:
        if len(out) > 0:
            sub_dfs = out
            out = []

        for sub in sub_dfs:
            pair = '' if 'clust' in ytype else '_pair'
            splitvar = x if len(reactivation_type) == 0 else '%s_%s_replay%s_plus_day1' % (
                reactivation_type.split('-')[0], reactivation_type.split('-')[1], pair
            )
            out.append(sub.loc[sub[splitvar] == 0, :])
            out.append(sub.loc[sub[splitvar] > 0, :])

    if values:
        if len(out) > 0:
            sub_dfs = out
            out = []
        else:
            sub_dfs = [sub]

        for sub in sub_dfs:
            xy = sub[[x, 'd_%s'%y]].dropna()
            out.append((xy[x].values, xy['d_%s'%y].values))

    return out


def _ranges(ytype):
    """
    Return the ranges for plotting.

    Parameters
    ----------
    ytype : str
        Can be 'noise', 'noise-nolick', 'clust', or 'spont'

    Returns
    -------
    List of two tuples for the x-range and y-range

    """

    if 'spont' in ytype:
        return [(-0.05, 0.05), (0, 70)]
    else:
        return [(-0.5, 0.5), (0, 7)]

def _reptype(ytype, group=None, specific=True, cs='plus'):
    """
    Return a replay type based on a ytype

    Parameters
    ----------
    ytype : str
        'clust' or not 'clust' will be cell-based or pair-based
    group : str
        Blank will be default y, or can be 'reward' or 'non-reward'
    specific : str
        Non-specific or specific if group is not empty
    cs : str
        Can only split into groups if cs is 'plus'

    Returns
    -------
    y name: str

    """

    if group is None or len(group) == 0:
        return 'repcount_%s_day1' % cs if 'clust' in ytype else 'reppair_%s_day1' % cs

    group = 'non' if group[:3] == 'non' else ''
    specificity = '' if specific else 'non'
    pair = '' if 'clust' in ytype else '_pair'
    return '%sreward_%sspecific_replay%s_%s_day1' % (group, specificity, pair, cs)


def _names(df, val=None, rep=None, cs='plus'):
    """
    Return the y axis name based on the value and dataframe

    Parameters
    ----------
    df : dataframe
    val : str
        'noise', 'noise-nolick', 'spont', 'clust'
    cs : str
        Stimulus

    Returns
    -------
    (type str, x-axis value str, y-axis value str)
        Will fit 'clust' or 'noise' by default

    """

    # Set defaults for pairs
    ctype = val if val is not None else 'noise'
    x = 'reppair_%s_day1' % cs
    y = 'noise_correlation_%s'%cs

    # Defaults for cell-specific analyses
    if 'reppair_%s_day1' % cs not in df.keys() and 'd_noise_correlation_%s' % cs not in df.keys():
        ctype = 'clust'
        x = 'repcount_%s_day1'%cs
        y = 'graph_clustering_%s'%cs

    # Adjust x/replay value if necessary
    if rep is not None:
        if rep == 'reward-specific':
            x = 'reward_specific_replay_pair_%s_day1' % cs
        elif rep == 'reward-nonspecific':
            x = 'reward_nonspecific_replay_pair_%s_day1' % cs
        elif rep == 'nonreward-specific':
            x = 'nonreward_specific_replay_pair_%s_day1' % cs
        elif rep == 'nonreward-nonspecific':
            x = 'nonreward_nonspecific_replay_pair_%s_day1' % cs
        else:
            x = rep

        if ctype == 'clust' and 'reward' in rep and 'specific' in rep:
            x = x.replace('pair_', '')

    # Adjust y/connection value
    if val is not None:
        if val == 'noise':
            y = 'noise_correlation_%s' % cs
        elif val == 'clust':
            y = 'graph_clustering_%s' % cs
        elif val == 'noise-nolick':
            y = 'noise_correlation_nolick_%s' % cs
        elif 'spont' in val and val[:5] == 'spont':
            y = 'spontcorr_allall'
        else:
            y = val

    return ctype, x, y


# Everything below here is terribly written. Should all be pulled out.
def combdrive(df, threshold=30):
    """
    Combined visual-drivenness dataframe across stimuli.

    Parameters
    ----------
    df : dataframe
    threshold : int, visually driven threshold

    Returns
    -------
    subsetted dataframe

    """

    return df.loc[((df['visually_driven_plus_day1'] >= threshold)
                  & (df['visually_driven_plus_day2'] >= threshold))
                  | ((df['visually_driven_neutral_day1'] >= threshold)
                  & (df['visually_driven_neutral_day2'] >= threshold))
                  | ((df['visually_driven_minus_day1'] >= threshold)
                  & (df['visually_driven_minus_day2'] >= threshold)), :]


def uniquenames(df):
    """
    Convert mice and days into unique indices for GLMMs

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    dataframe
    """

    mice = df['mouse_day1'].unique().tolist()
    df['mousenum'] = 0
    for i, mouse in enumerate(mice):
        df.loc[df['mouse_day1'] == mouse, 'mousenum'] = i
    return df


def uniquecells(df, n=10):
    """
    Add a column for unique cells for each day
    Parameters
    ----------
    df
    n

    Returns
    -------

    """

    df['cellnum'] = 0
    for day1 in df['date_day1'].unique():
        nums = np.arange(len(df.loc[df['date_day1'] == day1, :]))
        np.random.shuffle(nums)

        df.loc[df['date_day1'] == day1, 'cellnum'] = nums

    return df.loc[df['cellnum'] < n, :]


def coeff_baseline(formula, df, n=100, family='gaussian', link='identity'):
    """
    Apply a GLM using formula to the data in the dataframe data

    :param formula: text formulax
    :param df: pandas dataframe df
    :param family: string denoting what family of functions to use, gaussian, gamma, or poisson
    :param link: link function to use, identity or log
    :param dropzeros: replace zeros with nans if True
    :param r: use the R programming language's version if true
    :return: None
    """

    if link.lower() == 'log':
        linkfn = sm.families.links.log
    else:
        linkfn = sm.families.links.identity

    if family.lower() == 'gamma':
        family = sm.families.Gamma(link=linkfn)
    elif family.lower() == 'gaussian' or family == 'normal':
        family = sm.families.Gaussian(link=linkfn)
    else:
        family = sm.families.Poisson(link=linkfn)

    vals = []
    for i in range(n):
        # print(i)
        singlecell = uniquecells(df, 1)

        sub = flow.misc.math.subformula(formula, singlecell)
        sub.dropna(inplace=True)

        y, X = patsy.dmatrices(formula, sub, return_type='dataframe')

        model = sm.GLM(y, X, family=family)
        glm_results = model.fit()

        vals.append(glm_results.params[1])

    print('Mean coefficient +- stdev: %.4f +- %.5f' %
          (np.mean(vals), np.std(vals)))
    print('Number of coefficients greater than 0: %i/%i' %
          (np.sum(np.array(vals) > 0), n))


def compare_across_animals(y, sdf, reptype='reactivation', across='mouse'):
    """
    Compare a single value across all animals.

    Parameters
    ----------
    y : str
        Parameter name
    df : dataframe
    type : str {'reactivation', 'clusters', 'rich-poor'}
        The type of comparison being performed.

    """

    np.warnings.filterwarnings('ignore')
    comb = [[], []]
    gr = flow.grapher.graph('', 'half')

    if across == 'mouse':
        iterate_over = sdf.df['mouse_day1'].unique().tolist()
    else:
        iterate_over = sdf.df['date_day1'].unique().tolist()

    for mouse_day in iterate_over:
        if across == 'mouse':
            sdf.df = sdf.subset('plus', mouse=mouse_day)
        else:
            sdf.df = sdf.subset('plus')
            sdf.df = sdf.df.loc[sdf.df['date_day1'] == mouse_day, :]

        if reptype == 'reactivation':
            subs = sdf.split(reactivation=True)

            for i, df in enumerate(subs):
                vals = df[y].dropna().values
                comb[i].append(np.nanstd(vals))
        elif reptype == 'clusters':
            subs = sdf.split(reactivation=True, cluster=True)
            subs = subs[1, 3]

            for i, df in enumerate(subs):
                vals = df[y].dropna().values
                comb[i].append(np.nanmean(vals))
        elif reptype == 'rich-poor':
            subs1 = sdf.split(reactivation=True, reactivation_type='reward-specific')
            subs2 = sdf.split(reactivation=True, reactivation_type='nonreward-specific')
            subs = [subs2[1], subs1[1]]

            for i, df in enumerate(subs):
                vals = df[y].dropna().values
                comb[i].append(np.nanmean(vals))

        gr.add([0, 1], [comb[0][-1], comb[1][-1]], **{'opacity': 0.5, 'line-width':1.0})

    sdf.df = sdf.subset('plus')
    gr.add([0, 1], np.nanmean(comb, axis=1), **{'color': 'black'})
    gr.line(**{
        'ytitle':'Change in connectivity',
        'xtitle':reptype + ' across ' + across,
        'ymin': 0,
        'dots': True,
        'ymax': 0.3,
    })

    print('Means: %.4f %.4f,  >0 %i/%i,   Wilcoxon 2-tail signed-rank: %.4f,   2-tail ranksum: %.4f' % (
        np.nanmean(comb[0]), np.nanmean(comb[1]),
        np.nansum((np.array(comb[1]) - np.array(comb[0])) > 0), len(comb[0]),
        scipy.stats.wilcoxon(comb[0], comb[1]).pvalue,
        scipy.stats.ranksums(comb[0], comb[1]).pvalue,
    ))

