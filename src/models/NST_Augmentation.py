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

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

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

def resnet_layers(layer_names):
    base_model = ResNet50(include_top=False, weights='/workspace/storage/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224, 3))
    outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

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

@tf.function()
def train_step(image, extractor, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_and_preprocess_style_image(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float16)
    img = tf.image.resize(img, (224, 224))
    img = img[tf.newaxis, :]
    return img

def load_and_preprocess_content_image(path_to_img, style_image_shape):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float16)
    img = tf.image.resize(img, (style_image_shape[1], style_image_shape[2]))  # Resize to match style image shape
    img = img[tf.newaxis, :]
    return img

def lrp_backward_propagation(model, input_image, target_layer):
    # Preprocess the input image
    input_image = input_image * 255.0
    preprocessed_input = preprocess_input(tf.cast(input_image, tf.float16))

    # Get the model's layers
    layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)]

    # Forward pass to get activations
    activations = []
    x = preprocessed_input
    for layer in layers:
        x = layer(x)
        activations.append(x)

    # Find index of target layer
    target_layer_index = layers.index(model.get_layer(target_layer))

    # Start with the target layer
    R = activations[target_layer_index]

    # Backward propagation of relevance
    for i, layer in enumerate(reversed(layers[:target_layer_index])):
        activation = activations[target_layer_index - i - 1]
        if isinstance(layer, tf.keras.layers.Conv2D):
            W = layer.weights[0]
            W_pos = tf.maximum(W, 0)
            
            # Pad the activation to match the expected input shape
            padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])  # Adjust padding if necessary
            padded_activation = tf.pad(activation, padding, "CONSTANT")
            
            z = tf.nn.conv2d(padded_activation, W_pos, strides=layer.strides, padding='SAME') + layer.bias
            s = R / (z + tf.keras.backend.epsilon())
            
            # Calculate the correct output shape for the transposed convolution
            output_shape = tf.shape(activation) 
            
            c = tf.nn.conv2d_transpose(s, W_pos, output_shape=output_shape, strides=layer.strides, padding='SAME')
            R = activation * c
        elif isinstance(layer, tf.keras.layers.Dense):
            W = layer.weights[0]
            W_pos = tf.maximum(W, 0)
            z = tf.matmul(activation, W_pos) + layer.bias
            s = R / (z + tf.keras.backend.epsilon())
            c = tf.matmul(s, tf.transpose(W_pos))
            R = activation * c

    # Normalize the relevance scores
    R = R / tf.reduce_sum(R)

    return R

def extract_features_with_lrp(image, extractor, style_targets):
    """
    Extract features from all layers of the ResNet model using Layer-wise Relevance Propagation (LRP).
    """
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
        target_shape = (14, 14)  # Adjust the target shape to a smaller size
        content_features_resized.append(tf.image.resize(output, target_shape))

    # Concatenate the content features
    content_features_concat = tf.concat(content_features_resized, axis=-1)

    # Ensure correct dimensionality in the channel dimension
    num_channels = style_targets['conv4_block6_out'].shape[-1]  # Get the number of channels from the style targets
    content_features_concat = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=1, padding='valid')(content_features_concat)

    return content_features_concat

# Initialize Horovod
hvd.init()

# Configure TensorFlow to use only one GPU per process
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Function to broadcast variables
@tf.function
def broadcast_variables():
    hvd.broadcast_variables(opt.variables(), root_rank=0)
    hvd.broadcast_variables(extractor.variables, root_rank=0)

# Function for NST
def nst(content_images, style_images, content_layers, style_layers, opt):
    print("NST started")
    extractor = StyleContentModel(style_layers, list(content_layers.keys()))        

    style_targets = [extractor(style_img)['style'] for style_img in style_images]
    content_targets = [extractor(content_img)['content'] for content_img in content_images]
    
    # Distribute content images across GPUs
    local_content_images = np.array_split(content_images, hvd.size())[hvd.rank()]

    image = tf.Variable(local_content_images[0], trainable=True)

    style_weight = 1e-2
    content_weight = 1e4
    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    epochs_per_image = 50

    global augmented_images
    
    for content_idx, content_img in enumerate(local_content_images):
        for style_target in style_targets:
            image.assign(content_img)

            content_features = extract_features_with_lrp(content_img, extractor, style_target)

            for epoch in range(epochs_per_image):
                train_step(image, extractor, style_target, {'conv4_block6_out': content_features}, style_weight, content_weight, num_style_layers, num_content_layers)
                print(f"Content Image {content_idx + 1}/{len(local_content_images)}, Epoch {epoch+1}/{epochs_per_image}", end='\r')

            augmented_images.append(tf.image.resize(image, (224, 224)))
    
    print("NST done")
    
    return augmented_images

if __name__ == "__main__":
    t0 = time.time()
    print("Start time", t0)
    
    ## Provide content and style image data

    content_folder = ''
    style_folder = ''

    style_images = []
    for filename in os.listdir(style_folder):
        if filename.endswith(".jpg"):
            path_to_style_img = os.path.join(style_folder, filename)
            style_img = load_and_preprocess_style_image(path_to_style_img)
            style_images.append(style_img)

    content_images = []
    for filename in os.listdir(content_folder):
        if filename.endswith(".jpg"):
            path_to_content_img = os.path.join(content_folder, filename)
            content_img = load_and_preprocess_content_image(path_to_content_img, style_img.shape)
            content_images.append(content_img)

    print("data loaded\n")
    t1 = time.time()

    style_layers = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    content_layers = {'conv4_block6_out': 3}
    #opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)

    # Wrap the optimizer with Horovod's DistributedOptimizer
    #opt = hvd.DistributedOptimizer(opt) ## This configuration effective for small number of gpus
    
    
    opt = tf.keras.optimizers.Adam()
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True) ## This one is effective for multi gpus
    augmented_images = []
    augmented_images = nst(content_images, style_images, content_layers, style_layers, opt)

    t2 = time.time()
    print(" time", t2)

    final_images = [tf.cast(img * 255, tf.uint8) for img in augmented_images]
    final_images_resized = [tf.image.resize(img, (224, 224)) for img in final_images]
    final_images_uint8 = [tf.cast(img, tf.uint8) for img in final_images_resized]
    ## Provide the output folder
    output_folder = ''
    os.makedirs(output_folder, exist_ok=True)
    import threading

    def save_image_async(image, path):
        threading.Thread(target=lambda: tf.io.write_file(path, tf.image.encode_jpeg(tf.squeeze(image)))).start()
    
    # Use this function inside the main saving loop
    for idx, img in enumerate(final_images_uint8):
        filename = f"augmented_image_{idx + 1}.jpg"
        filepath = os.path.join(output_folder, filename)
        
        save_image_async(img, filepath)
        print(f"Saving {filepath} asynchronously...")


    t3 = time.time()
    print("elapsed time", t3 - t0)

    print("\nAugmented images saved successfully.")
