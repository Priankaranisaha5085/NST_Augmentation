import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow import keras
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D
import PIL


### Clip pixel values to the range [0, 1]
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

###Calculate the Gram matrix for style images
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

###Style and Content Model using ResNet50 for extracting features from style and content images

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.resnet = resnet_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.resnet.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.resnet(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
    
    
###Create a ResNet model with specified style and content layer outputs 

def resnet_layers(layer_names):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(inputs=base_model.input, outputs=outputs)
    return model


### This function calculate losses for training Resnet50 to perform style transfer
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss


###Training step function
### Iteratively optimize the initial image based on the style and content loss    
###Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


### Load and preprocess style image
def load_and_preprocess_style_image(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (##Resize the image to match your dataset size)) 
    img = img[tf.newaxis, :]
    return img


### Load and preprocess content image
def load_and_preprocess_content_image(path_to_img, style_image_shape):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (style_image_shape[1], style_image_shape[2]))  # Resize to match style image shape
    img = img[tf.newaxis, :]
    return img


### Extract features from content images with Layer-wise Relevance Propagation (LRP)
def extract_features_with_lrp(image, extractor, style_targets):
    # Preprocess the image
    image = image * 255.0
    preprocessed_input = preprocess_input(image)

    # Get all layer outputs
    outputs = extractor.resnet(preprocessed_input)

    # Resize content features to match content targets
    content_features_resized = []
    for layer_name in content_layers:
        output_index = extractor.resnet.output_names.index(layer_name)
        output = outputs[output_index]
        target_shape = (x, y)  # Adjust the target shape (x,y) to a smaller size
        content_features_resized.append(tf.image.resize(output, target_shape))

    # Concatenate the content features
    content_features_concat = tf.concat(content_features_resized, axis=-1)


### Initialize Horovod for parallelization
hvd.init()

### Configure TensorFlow to use only one GPU per process
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

### Function to broadcast variables
@tf.function
def broadcast_variables():
    hvd.broadcast_variables(opt.variables(), root_rank=0)

### Neural Style Transfer (NST) function
def nst(content_images, style_images, content_layers, style_layers, opt):
    print("NST started")
    extractor = StyleContentModel(style_layers, list(content_layers.keys()))        

    style_targets = [extractor(style_img)['style'] for style_img in style_images]
    content_targets = [extractor(content_img)['content'] for content_img in content_images]
    
    ### Distribute content images across GPUs
    local_content_images = np.array_split(content_images, hvd.size())[hvd.rank()]

    image = tf.Variable(local_content_images[0], trainable=True)

    style_weight = 1e-2
    content_weight = 1e4
    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    epochs_per_image = # set the epoch

    global augmented_images
    
    for content_idx, content_img in enumerate(local_content_images):
        for style_target in style_targets:
            image.assign(content_img)

            content_features = extract_features_with_lrp(content_img, extractor, style_target)

    return augmented_images

### Measuring Time of NST_Augmentation
if __name__ == "__main__":
    t0 = time.time()
    print("Start time", t0)
    
    ### Data loading
    content_folder = 'Give the folder path for the content image'
    style_folder = 'Give the folder path for the style image'
    
    
    style_images = []
    for filename in os.listdir(style_folder):
        if filename.endswith("Set the image file extention (ex: .jpg, .png, etc.)"):
            path_to_style_img = os.path.join(style_folder, filename)
            style_img = load_and_preprocess_style_image(path_to_style_img)
            style_images.append(style_img)

    content_images = []
    for filename in os.listdir(content_folder):
        if filename.endswith("Set the image file extention (ex: .jpg, .png, etc.)"):
            path_to_content_img = os.path.join(content_folder, filename)
            content_img = load_and_preprocess_content_image(path_to_content_img, style_img.shape)
            content_images.append(content_img)

    print("data loaded\n")
    t1 = time.time()

    style_layers = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    content_layers = {'conv4_block6_out': 3}
    opt = tf.keras.optimizers.Adam(learning_rate=# set the learning rate, beta_1=# set the beta value in between 0 to 1, epsilon=1e-# set the epsilon value in between 1e-1 to 1e-2)

    ### Wrap the optimizer with Horovod's DistributedOptimizer
    opt = hvd.DistributedOptimizer(opt)
    
    augmented_images = []
    augmented_images = nst(content_images, style_images, content_layers, style_layers, opt)

    t2 = time.time()
    print(" time", t2)

    final_images = [tf.cast(img * 255, tf.uint8) for img in augmented_images]
    final_images_resized = [tf.image.resize(img, (##Resize the image to match your dataset size)) for img in final_images]
    final_images_uint8 = [tf.cast(img, tf.uint8) for img in final_images_resized]
    
    ###Saving the Augmented image
    output_folder = 'Give the folder path for storing the output'
    
    os.makedirs(output_folder, exist_ok=True)
    for idx, img in enumerate(final_images_uint8):
        filename = f"augmented_image_{idx + 1}.jpg"
        filepath = os.path.join(output_folder, filename)

        img = tf.squeeze(img, axis=0)
        encoded_img = tf.image.encode_jpeg(img)
        tf.io.write_file(filepath, encoded_img)
        print(f"Saved {filepath}")

    t3 = time.time()
    print("elapsed time", t3 - t0)

    print("\nAugmented images saved successfully.")
