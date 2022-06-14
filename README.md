Scripts for unpacking ana analyzing data collected with LinoSPAD2.

The 'main' is the main hub where individual modules are called.
Following modules can be used:

    * cross_talk - calculation of the cross-talk rate
    * cross_talk_plot - plots the cross-talk rate distribution in the
    LinoSPAD2 pixels
    * cross_talk_fast - 4-times faster script for calcultion of the cross-talk
    rate that does not work with the pixel coordinates
    * differences - calculation of the differences between all timestamps
    which can be used to calculate the Hanbury-Brown and Twiss peaks
    * td_plot - plot a histogram of timestamp differences from LinoSPAD2

