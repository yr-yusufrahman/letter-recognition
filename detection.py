import numpy as np
import cv2
import string
import matplotlib.pyplot as plt

def isolate(name):
    image = cv2.imread(name)
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to greyscale
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    # Invert colours
    inverted = cv2.bitwise_not(grey)
    
    # Binary image
    _, binary = cv2.threshold(inverted, 155, 255, cv2.THRESH_BINARY) 
    cv2.floodFill(binary, None, (0, 0), 0)
    
    # Create a dilated image
    kernel = np.ones((5, 8), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations = 1)
    
    # Find contours on dilated binary image
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed = []
    # Draw all the found contours onto the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=1)
        
        # Cropping out the letter from the image corresponding to the current contours
        letter = binary[y:y+h, x:x+w]
        
        # Resize image
        resize_factor = 64
        resized = cv2.resize(letter, (resize_factor, resize_factor))
    
        # Adding the preprocessed digit to the list of preprocessed digits
        processed.append(resized)

    return processed, original, contours


def format(size, processed):
    size = size**2
    pre_input = []
    for i in range(len(processed)):
        flat = processed[i].flatten()
        pre_input.append(flat)
    
    pre_input = np.vstack(pre_input)
    return pre_input


def filtering(input, filter):
    n = input.shape[0]
    features = input.shape[1]
    image_size = int(features**0.5)
    filter_size = filter.shape[1]

    mini = []
    
    # Identifies which indices need to be accessed - 3x3 grid
    location = np.arange(image_size**2).reshape((image_size, image_size))
    for row in range(1, image_size-1):
        for column in range(1, image_size-1):
            filter_grid = location[row-1:row+2, column-1:column+2].flatten()
            grid = input[:, filter_grid]
            mean = grid.dot(filter.T) / filter_size
            mini.append(mean)

    bitmap = np.hstack(mini)      
    return bitmap


def relu(Z):
    return np.maximum(0, Z)


def pooling(input):
    # 2x2 kernel
    pool_kernel = 2
    features = input.shape[1]
    pool_size = int(features**0.5)

    mini = []
    
    # Identifies which indices need to be accessed - 2x2 grid
    location = np.arange(features).reshape((pool_size, pool_size))
    for row in range(0, pool_size, pool_kernel):
        for column in range(0, pool_size, pool_kernel):
            pool_grid = location[row:row+pool_kernel, column:column+pool_kernel].flatten()

            # Get the maximum value from each row in the matrix
            maximum = input[:, pool_grid].max(1).reshape((-1, 1))
            mini.append(maximum.round())
    
    result = np.hstack(mini) 
    return result


def process(images, filter):
    bitmap = filtering(images, filter)
    adjusted = relu(bitmap)
    result = pooling(adjusted)
    return result


def convolution(images):
    all = []
    # Vertical filter
    vertical = np.array([[1, 0, -1, 1, 0, -1, 1, 0, -1]])  
    result = process(images, vertical)
    all.append(result)
    
    # Horizontal filter
    horizontal = np.array([[1, 1, 1, 0, 0, 0, -1, -1, -1]]) 
    result1 = process(images, horizontal)
    all.append(result1)

    # Diagonal filter
    diagonal = np.array([[1, 1, 0, 1, 0, -1, 0, -1, -1]])  
    result2 = process(images, diagonal)
    all.append(result2)
    
    final = np.hstack(all)
    return final


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def feed(model, X):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    W3 = model['W3']
    b3 = model['b3']

    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    
    return A3


def display(name, predictions, y_hat):
    alphabet = string.ascii_uppercase
    processed, original, contours = isolate(name)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        probability = str(round(y_hat[:, i].max() * 100, 2)) + '%'
        predicted = alphabet[predictions[i]] + ': ' + probability

        # Add a title above the contour
        cv2.putText(original, predicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 155, 0), 1)

    # Save image with contours
    drawn = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite('pi-result-HD.png', drawn)
    plt.imshow(original)
    plt.title('Predictions')
    plt.show()

file = 'image.jpg'
# The binary image is blank
processed, original, contours = isolate(file)
pre_input = format(64, processed)
input = convolution(pre_input).T / 255.0

# Load matrices from the file
modelName = 'parameters.npz'
model = np.load(modelName)

y_hat = feed(model, input)
predictions = np.argmax(y_hat, 0)
display(file, predictions, y_hat)
