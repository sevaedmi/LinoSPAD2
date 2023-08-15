def compute_delta_t(pixel_0, pixel_1, timestampsnmr: int = 512, timewindow: int = 5000):
    nmr_of_cycles = int(len(pixel_0) / timestampsnmr)
    output = []

    # start = time.time()
    for cycle in range(nmr_of_cycles):
        for timestamp_pix0 in range(timestampsnmr):
            if (
                pixel_0[cycle * timestampsnmr + timestamp_pix0] == -1
                or pixel_0[cycle * timestampsnmr + timestamp_pix0] <= 1e-9
            ):
                break
            for timestamp_pix1 in range(timestampsnmr):
                if (
                    pixel_1[cycle * timestampsnmr + timestamp_pix1] == -1
                    or pixel_1[cycle * timestampsnmr + timestamp_pix1] == 0
                ):
                    break
                if (
                    np.abs(
                        pixel_0[cycle * timestampsnmr + timestamp_pix0]
                        - pixel_1[cycle * timestampsnmr + timestamp_pix1]
                    )
                    < timewindow
                ):
                    output.append(
                        pixel_0[cycle * timestampsnmr + timestamp_pix0]
                        - pixel_1[cycle * timestampsnmr + timestamp_pix1]
                    )
                else:
                    continue
    return output
