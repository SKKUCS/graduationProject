
from PIL import Image
    
    img = Image.fromarray(state, 'RGB')
    img.save('my.png')
    img.show()
