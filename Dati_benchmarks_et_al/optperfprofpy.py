import warnings

import numpy as np
import pandas as pd
from math import isclose
import matplotlib.pyplot as plt


def calc_perprof(df, problem_def, perf_meas, solver_char, inv_perf_meas=False, tau_val=None, safe_divisions=False):
    """Generate array for performance profiles.

    Notes
    -----
    For a detailed description of performance profiles see the publication:
    Benchmarking Optimization Software with Performance Profiles by
    E. D. Dolan, and J. J. More.

    Parameters
    ----------
    df : pandas DataFrame
        Data containing the problem definition, performance measure,
        and solver characteristics.
    problem_def : list
        Label that define the unique problems.
        The use of multiple labels is supported.
    perf_meas : list containing string
        Label that indicates the performance measure.
    solver_char : list
        Label that defines the unique solvers.
        The use of multiple labels is supported.
    inv_perf_meas : bool, optional )defaults to False)
        Indicating if the assigned performance measure is the 
        value divided by the smallest value (standard), or the
        inverse of this operation.
    tau_val : numpy.ndarray, optional (defaults to None)
        If supplied, the number of problems the unique solvers
        have solved are checked at these values of tau.
    safe_divisions : bool, optional (defaults to False)
        Whether to divide the performance measure by its minimum value 
        according to the following extra rules: `0/0=1`, `Inf/0=Inf`, 
        `x/0=inf` 
        This may make sense, for example, when the best value is 0, 
        and another solver achieves 0 as well.

    Returns
    -------
    unique_taus : numpy.ndarray
        Unique tau values where the solvers have an increased
        number of solved problems. In this manner all information
        present is extracted.
    solver_taus : numpy.ndarray
        Number of problems each solver solved within the ratio
        of unique_taus.
    solvers : string
        Unique solver names
    data : pandas DataFrame
        Dataframe where the performance measure values have been
        normalized.

    Raises
    ------
    TypeError
        If any of the arguments is not a list.
    ValueError
        If the solver characteristics and problem definition
        share a string.
        Of if not all problems have been solved by the 
        unique solvers.
    ValueError
        If the problem lenghts are not e

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.

    """
    data = df.sort_values(by=problem_def + solver_char).copy()

    if not all(isinstance(l, list) for l in [problem_def, perf_meas, solver_char]):
        raise TypeError('`problem_def`, `perf_meas`, and `solver_char` should'
                        'be lists')

    if len(solver_char) > 1:
        # Merging columns if one than one solver characteristic is selected
        new_solver_nm = ''
        for n, m in enumerate(solver_char):
            if n == 0:
                new_solver_nm += data['{}'.format(solver_char[n])].map(str)
            else:
                new_solver_nm += '_' + data['{}'.format(solver_char[n])].map(str)
        
        data['{}'.format(solver_char[0])] = new_solver_nm

    if len(set(solver_char) & set(problem_def)) != 0:
        # Checking if problem definition and solver characteristic are unique
        raise ValueError('Solver characteristic and problem definition share characteristic: ',
                         list(set(solver_char) & set(problem_def)))

    # Finding the unique solvers
    solvers = data[solver_char[0]].unique()

    # Generating df containing all unique problems
    grouped_by_problem = data.groupby(problem_def)
    
    # dividing by the minimum value
    for i, (prob, gr) in enumerate(grouped_by_problem):
        # Checking if all problems have an equal number of solvers
        if i == 0:
            gr_len = len(gr)

        if gr_len != len(gr):
            raise ValueError('Problem group lengths not equal! Problem gr:', prob)

        try:
            def _safe_division(num, den):
                if den == 0:
                    if num == 0:    #0/0 = 1
                        return 1.
                    elif not np.isfinite(num):  # Inf/0 = Inf
                        return np.inf
                    else:
                        return np.inf   # x/0 = Inf
                return num / den

            # Normalizing and penalizing infeasible designs
            # If feasibility is satisfied, the performance measure is compared to
            # the minimum value among all methods that are feasible.

            # If feasibility is not satisfied, the maximum occuring value among
            # all solvers is allocated and a small value is added.
            # This value is added to be able to differentiate between the solvers
            # that terminated with the maximum value that are feasible from
            # the solvers that did not return a feasible point
            true_min = gr.loc[gr['feas'] == True][perf_meas].min().iloc[0]
            if inv_perf_meas == False:
                data.loc[gr.loc[gr['feas'] == True].index, perf_meas] = \
                    gr[perf_meas].map(lambda x: _safe_division(x, true_min)) if safe_divisions else gr[perf_meas] / true_min
                data.loc[gr.loc[gr['feas'] == False].index, perf_meas] = \
                    gr[perf_meas].map(lambda x: _safe_division(x, true_min)) + .05 if safe_divisions else gr[perf_meas] / true_min +.05
            
            else:
                if i == 0:
                    warnings.warn('Performance ratio calculated using inverse.')
                data.loc[gr.loc[gr['feas'] == True].index, perf_meas] = \
                    gr[perf_meas].map(lambda x: _safe_division(true_min, x)) if safe_divisions \
                                    else true_min / gr[perf_meas]
                data.loc[gr.loc[gr['feas'] == False].index, perf_meas] = \
                    gr[perf_meas].map(lambda x: _safe_division(true_min, x)) + .05 if safe_divisions \
                                    else true_min / gr[perf_meas] +.05

            # if any(data.loc[gr.loc[gr['feas'] == True].index, perf_meas] > 1e2):
            #     print(f"{gr["kind"]}")
            #     print(f"{data.loc[gr.loc[gr['feas'] == True].index, perf_meas]}")
            #     print()
        except KeyError:
            if not inv_perf_meas:
                data.loc[gr.index, perf_meas] = gr[perf_meas] / gr[perf_meas].min().iloc[0]
            else:
                if i == 0:
                    warnings.warn('Performance ratio calculated using inverse.')
                data.loc[gr.index, perf_meas] = gr[perf_meas].min().iloc[0] / gr[perf_meas]

    # Generate array for plot
    if (df[perf_meas[0]] < 0).any():
        warnings.warn('Negative objective function value detected, this may '
                      'cause unwanted scaling of problems.')

    if (len(data) // len(solvers)) != len(grouped_by_problem):
        warnings.warn('Combination of problem and solver characteristic '
                      'cause, possibly unwanted, aggregation of problems.')

    # Grouping by unique solver
    grouped_by_solver = data.groupby(solver_char)

    if tau_val == None:
        # Finding the unique tau values
        unique_taus = np.sort(data[perf_meas[0]].unique())
    else:
        # Using the user generated tau values
        unique_taus = tau_val

    # Finding the fraction of problems that each solver solved within tau
    solver_taus = {solver: np.zeros(len(unique_taus)) for solver in solvers}
    for n, tau in enumerate(unique_taus):
        for i, (solver, gr) in enumerate(grouped_by_solver):
            solver_key = solver[0] #dovrebbe bastare perché la prima colonna è sovrascritta col nome completo (?)
            if i == 0 and n == 0:
                print('Number of problems per solver: ', len(gr))
            solver_taus[solver_key][n] = len(gr.loc[gr[perf_meas[0]] <= tau]) / len(grouped_by_problem)
            
    solved_at_first_tau = sum(vals[0] for vals in solver_taus.values())
    if not isclose(solved_at_first_tau, 1, rel_tol=1e-3):
        warnings.warn('Solvers do not solve 100% of problems. '
                      'Total amount of problems solved: {}'.format(100 * solved_at_first_tau))

    return unique_taus, solver_taus, solvers, data

def draw_simple_pp(taus, solver_vals, solvers):
    """Simple step plotter for performance profiles.

    Parameters
    ----------
    taus : numpy.ndarray
        x values of plot.
    solver_vals : numpy.ndarray
        y values of plot.
    solvers : list
        Labels of curves.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Add lines individually to support labels
    for n, solver in enumerate(solvers):
        ax.step(taus, solver_vals[n, :], where='post', label=solver)
        
    plt.legend(loc=4)
    plt.xlim(1, max(taus))
    ax.set_xlabel('Tau')
    ax.set_ylabel('Fraction of problems')
    
    plt.plot()
