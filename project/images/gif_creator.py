from PIL import Image

gif_name = 'swiss_roll.gif'

image_filenames = []
for i in range(0,60):
    filename = 'swiss_roll/swissroll_' + str(i) + '_iterations.png'
    image_filenames.append(filename)

frames = []
for img_file in image_filenames:
    img = Image.open(img_file)
    frames.append(img)

frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=150, loop=0)
