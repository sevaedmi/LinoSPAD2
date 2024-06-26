from line_profiler import LineProfiler


def the_function():
    # do something
    pass


def profile_function():
    lp = LineProfiler()
    lp.add_function(the_function)
    lp_wrapper = lp(the_function)
    lp_wrapper()
    lp.print_stats()
