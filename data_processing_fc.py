import tensorflow as tf

def _prepare_data_fn(features, input_shape, augment=False):
    """Data from tfds. Resize image to expected dimensions, and 
    opt. apply some random transformations.
    
    Arguments:
        features {[Array]} -- [Data]
        input_shape {[Tuple]} -- [Shape, usually (h, w, c), expected by the models (images will be resized 
        accordingly)]
    
    Keyword Arguments:
        augment {bool} -- [Flag to apply some random augmentations to the images] 
        (default: {False})
    
    Returns:
        [type] -- [Augmented Images, Labels]
    """    
    input_shape = tf.convert_to_tensor(input_shape)
    
    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples:
    image = features['image']
    label = features['label']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    if augment:
        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0) # keeping pixel values in check

        # Random resize and random crop back to expected size:
        
        random_scale_factor = tf.random.uniform([1], minval=1., maxval=1.4, dtype=tf.float32)
        scaled_height = tf.cast(tf.cast(input_shape[0], tf.float32) * random_scale_factor, 
                                tf.int32)
        scaled_width = tf.cast(tf.cast(input_shape[1], tf.float32) * random_scale_factor, 
                               tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, input_shape)
    else:
        image = tf.image.resize(image, input_shape[:2])
        
    return image, label