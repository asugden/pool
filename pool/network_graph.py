from flow.netcom import NCGraph


def graph(andb, corrtype, extralimits=None, replacelimits=False, vdrive=50):
    """
    Create a new instance of a networkx graph of a particular correlation type
    :param andb: analysis database with mouse and date set
    :param corrtype: correlation type, str
    :param extralimits: a boolean array or int array of which cells to include, combined with that given by corrtype
    :param replacelimits: if True, replace the current limits with extralimits
    :param vdrive: visual-drivenness inverse p value
    :return: instance of NCGraph
    """

    if corrtype == 'spontaneous':
        corr = andb['spontaneous-correlation']
        corr[corr > 0.0038 + 3*0.0188] = np.nan
        limits = None
    else:
        corr = andb['noise-correlation-%s' % corrtype]
        limits = andb['visually-driven-%s' % corrtype.replace('nolick-', '')] > vdrive

    ncells = np.shape(corr)[0]

    if replacelimits or limits is None:
        limits = extralimits
    elif extralimits is not None:
        limits = np.array(limits)
        extralimits = np.array(extralimits)

        if limits.dtype == np.bool and extralimits.dtype == np.bool:
            limits = np.bitwise_and(limits, extralimits)
        elif limits.dtype != np.bool and extralimits.dtype != np.bool:
            limits = np.union1d(limits, extralimits)
        elif limits.dtype == np.bool:
            comb = np.zeros(len(limits)) > 1
            comb[extralimits] = True
            limits = np.bitwise_and(limits, comb)
        else:
            comb = np.zeros(len(extralimits)) > 1
            comb[limits] = True
            limits = np.bitwise_and(comb, extralimits)

    out = NCGraph(ncells, corr, limits)
    return out
