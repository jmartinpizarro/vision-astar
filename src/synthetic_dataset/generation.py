from utils import *

def main():
    # generate img
    for grid_size in range(3,11,1):
        for i in range(1, NUM_IMAGES + 1):
            grid = generate_random_grid(grid_size)
            clean_img = draw_grid(grid, grid_size)
            noisy_img = apply_noise(clean_img)
            filename, matrix = save_grid_and_matrix(global_counter, grid, noisy_img)
            csv_lines.append(f"{filename},\"{matrix}\"")
            global_counter += 1

    # save csv
    with open(os.path.join(OUTPUT_DIR, "dataset.csv"), "w") as f:
        f.write("\n".join(csv_lines))

    print(f"Generated {NUM_IMAGES * 8} images with noise in: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
