from PIL import Image, ImageDraw, ImageFont
import os

size = 1024
img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Gradient background
for y in range(size):
    r = int(240 - (y / size) * (240 - 255))
    g = int(70 + (y / size) * (141 - 70))
    b = int(133 + (y / size) * (77 - 133))
    draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

# Draw text
try:
    font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 700)
except:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 700)
    except:
        font = ImageFont.load_default()

text = "字"
# Get text bounding box
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (size - text_width) / 2
y = (size - text_height) / 2 - 100 # Adjust vertically for Chinese characters

draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

# Mask for rounded corners
mask = Image.new('L', (size, size), 0)
draw_mask = ImageDraw.Draw(mask)
radius = 224 # Apple standard radius for 1024x1024 is ~22.5%
draw_mask.rounded_rectangle([(0, 0), (size, size)], radius=radius, fill=255)

img.putalpha(mask)
img.save('icon_1024.png')
print("icon_1024.png created.")
