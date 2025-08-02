import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# Parameters
IMAGE_SIZE = 300 # 300x300 px
CELL_SIZE = IMAGE_SIZE // 3
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" # change if you are using windows or macOS
FONT_SIZE = 60
FONT_SIZE_V2 = 48
OUTPUT_DIR = "data"
NUM_IMAGES = 500 # each grid size will have this num of images
csv_lines = ["filename,matrix"]
# make sure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
# possible options per cell: empty, origin, goal, obstacle
SYMBOLS = ["", "O", "G", "X"]

# Global counter to ensure unique filenames
global_counter = 1

def generate_random_grid(grid_size:int):
    """Generate a random AxB grid with 1 origin, 1 goal and up to 3 obstacles"""
    grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]
    # generate all possible positions, then suffle to add randomness
    positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(positions)
    # first two positions will be for the goal and the origin of the agent
    f_pos = positions.pop()
    g_pos = positions.pop()
    grid[f_pos[0]][f_pos[1]] = "O"
    grid[g_pos[0]][g_pos[1]] = "G"
    num_x = random.randint(0, 3) # number of obstacles
    for _ in range(num_x):
        if positions:
            x_pos = positions.pop()
            grid[x_pos[0]][x_pos[1]] = "X"
    return grid

def draw_grid(grid, grid_size:int):
    """Draws the grid in a PIL image"""
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw = ImageDraw.Draw(img)
    if grid_size <= 5:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    else:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE_V2)
        
    # draw lines of the grid
    cell_size = IMAGE_SIZE // grid_size
    for i in range(1, grid_size):
        draw.line([(0, i * cell_size), (IMAGE_SIZE, i * cell_size)], fill="black", width=3)
        draw.line([(i * cell_size, 0), (i * cell_size, IMAGE_SIZE)], fill="black", width=3)
    # draw letters
    for i in range(grid_size):
        for j in range(grid_size):
            symbol = grid[i][j]
            if symbol:
                _, _, w, h = draw.textbbox((0, 0), text=symbol, font=font)
                x = j * cell_size + (cell_size - w) // 2
                y = i * cell_size + (cell_size - h) // 2
                draw.text((x, y), symbol, fill="black", font=font)
    return img

def apply_noise(pil_img):
    """Applies realist distorsion to the image: noise, shadow, rotation and blur"""
    # convert to opencv
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    # dark shadows, simulating person doing photo
    rows, cols, _ = noisy_img.shape
    shadow = np.tile(np.linspace(1.0, 0.8, cols), (rows, 1))
    shadow = np.stack([shadow]*3, axis=2)
    noisy_img = (noisy_img * shadow).astype(np.uint8)
    # to PIL again
    noisy_pil = Image.fromarray(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    # light blur
    noisy_pil = noisy_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    # light rotation
    angle = random.uniform(-3, 3)
    noisy_pil = noisy_pil.rotate(angle, expand=False, fillcolor="white")
    return noisy_pil

def save_grid_and_matrix(idx, grid, img):
    filename = f"grid_{idx:04d}.jpeg"
    img.save(os.path.join(OUTPUT_DIR,'images',filename))
    return filename, grid

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