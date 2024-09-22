import torchvision
from PIL import Image
class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.vit = torchvision.models.vit_b_16(pretrained=True)
        self.cnn = nn.Conv2d
        self.fc = nn.Linear(768, 10)


    def forward(self, image):
        return self.vit(image)
    
    image_path = ''
    
    def process_image(self, image):
        img_path = image_path
        image = Image.open(img_path)
        image_data = np.array(image) 
        image = image / 255.0
        image = image - image.mean(dim=(1, 2, 3), keepdim=True)
        image = image / image.std(dim=(1, 2, 3), keepdim=True)
        return image, image_data

    def label_image(self, image, labels):
        focus_box = self.vit(image).focus_box
        focus_box(image_data)
        obj = focus_box
        for obj in image:
            return image, labels
    
    def train(self, image, image_path): 
        for image in image_path:
            self.process_image
            self.label_image
            self(image)
    
vision = VisionTransformer()
image_path = 'C:/users/jonny/Documents/PATH/ONI/knowledge_base/picturememory/'

