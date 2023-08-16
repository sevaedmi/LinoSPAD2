"""The main hub for analysis of LinoSPAD2 data.

"""


from functions import plot_valid

# ======================================================================
# Paths to where data or the 'csv' files with the resuts are located.
# ======================================================================

path_expl = ""


# ======================================================================
# Function execution.
# ======================================================================

if __name__ == "__main__":
    plot_valid.plot_valid_FW2212(
        path_expl,
        board_number="A5",
        fw_ver="2212b",
        timestamps=200,
        show_fig=True,
        app_mask=False,
    )
