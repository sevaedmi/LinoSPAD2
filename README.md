Scripts for unpacking and analyzing data collected with LinoSPAD2. The data
should be binary (the ".dat" format) as the ".txt" format takes much longer
to unpack. The 1D binary streams of data from the detector can be unpacked
and saved with the "unpack" module in the "functions" directory; the output
is a 2D matrix, where the columns cover the array of pixels (256 total) and
the rows are data values from each pixel.

The "main" is the main hub where individual modules are called. Modules for
real time plotting of the number of valid timestamps vs the pixel index
("plot_valid"), for plotting a grid of differences in timestamps for a given
range of pixels ("delta_t"), and for fitting the timestamp differences with
a gaussian function are available ("fits").

Additionally, an application with a GUI is available in the "app" directory.
Currently, two functions are available in the application: real-time plotting
and single pixel histogram, the latter can be used to check the data
streams for abnormalities.

The "tools" directory contains numerous scripts, some of which mirror some of
the modules in the "functions" as a standalone scripts that can be used as a
debugging tool. Other scripts were used in the testing of new functions.
