"""The main hub for analysis of data from LinoSPAD2.

"""


from functions import plot_valid

# =========================================================================
# Paths to where either data or the 'csv' files with the resuts are located.
# =========================================================================

path_JJ = "D:/LinoSPAD2/Data/BNL_Jesse_Joe/8-7-23/8-7-23"


# =========================================================================
# Function execution.
# =========================================================================

if __name__ == "__main__":
    plot_valid.plot_valid_FW2212(
        path_JJ,
        board_number="A5",
        fw_ver="2212b",
        timestamps=200,
        show_fig=True,
        app_mask=False,
    )
