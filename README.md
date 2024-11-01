## Project Goal
The goal of this project is to develop an Image Caption Generator that automatically generates descriptive captions for any given image. By combining computer vision and natural language processing techniques, the model interprets the content of an image and provides meaningful text descriptions, enhancing applications such as content tagging, accessibility, and visual search.

### Project Overview
+ **Dataset**
	+ The project uses the Flickr 8K dataset, which includes 8,000 images and their corresponding captions. The dataset is essential for training the model to understand the relationship between image features and textual descriptions.

+ **Feature Extraction**
	+ We utilize the pre-trained VGG16 model as a feature extractor. The fully connected layers of VGG16 are removed, and we use the penultimate layer’s output as the feature vector for each image.

	+ The image data is preprocessed, and feature vectors for all images are generated and stored in a serialized file (features.pkl). This preprocessing step reduces the complexity of image data and makes it suitable for further processing.

+ **Model Architecture**
The model for generating captions consists of a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN)
	+ **CNN**: VGG16 extracts image features, which are then fed into a dense layer to reduce dimensionality and introduce non-linearity.

	+ **RNN**: An LSTM network processes the image captions. The input captions are masked to handle varying lengths explicitly, and embedding layers are used for better word representation.
	+ **Decoder**: The CNN and RNN outputs are merged and passed through dense layers to generate the final caption.

+ **Training**
	+ The training process involves optimizing the model to learn the mapping between image features and the corresponding captions. Various techniques, such as dropout and embedding layers, are employed to improve performance and handle overfitting.

+ **Generating Captions**
	+ Once the model is trained, it can generate captions for new images. The image features are extracted using VGG16, and the trained model generates a descriptive caption, demonstrating its understanding of the visual content.
