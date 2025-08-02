import os
import utils


def main():
    # generate img
    for grid_size in range(3, 11, 1):
        for i in range(1, utils.NUM_IMAGES + 1):
            grid = utils.generate_random_grid(grid_size)
            clean_img = utils.draw_grid(grid, grid_size)
            noisy_img = utils.apply_noise(clean_img)
            filename, matrix = utils.save_grid_and_matrix(
                utils.global_counter, grid, noisy_img
            )
            utils.csv_lines.append(f'{filename},"{matrix}"')
            utils.global_counter += 1

    # save csv
    with open(os.path.join(utils.OUTPUT_DIR, "dataset.csv"), "w") as f:
        f.write("\n".join(utils.csv_lines))

    print(f"Generated {utils.NUM_IMAGES * 8} images with noise in: {utils.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
