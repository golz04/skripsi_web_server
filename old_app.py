from flask import Flask, render_template, request, jsonify
import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Start Route
@app.route("/")
def dashboard():
    dataset_dir = "static/dataset"

    image_files = []
    labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    image_files.append(image_path)
                    labels.append(class_name)

    X_rgb = []
    X_hsv = []
    X_gray = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (32, 32))

        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        flattened_rgb = rgb_image.flatten()
        X_rgb.append(flattened_rgb)

        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        flattened_hsv = hsv_image.flatten()
        X_hsv.append(flattened_hsv)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        flattened_gray = gray_image.flatten()
        X_gray.append(flattened_gray)

    label_mapping = {label: i for i, label in enumerate(set(labels))}
    y = np.array([label_mapping[label] for label in labels])

    X_train_rgb, X_test_rgb, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)
    X_train_hsv, X_test_hsv, y_train, y_test = train_test_split(X_hsv, y, test_size=0.2, random_state=42)
    X_train_gray, X_test_gray, y_train, y_test = train_test_split(X_gray, y, test_size=0.2, random_state=42)

    X_train = np.concatenate((X_train_rgb, X_train_hsv, X_train_gray), axis=1)
    X_test = np.concatenate((X_test_rgb, X_test_hsv, X_test_gray), axis=1)

    k = 4
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    custom_label_mapping = {i: label for i, label in enumerate(label_mapping.keys())}
    knn_model_custom = {
        "model": knn,
        "label_mapping": custom_label_mapping
    }

    with open("static/save_model/knn_model.pkl", "wb") as f:
        pickle.dump(knn_model_custom, f)

    dir_path = "static/save_model/"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "knn_model.h5")
    joblib.dump(knn, file_path)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    pageOn = 'dashboard'
    akurasi = accuracy*100
    return render_template('dashboard.html', getAccuracy=akurasi, pageOn=pageOn)

@app.route("/dataset/")
def dataset():
    dataset_dir = "./../dataset_nitrogen"
    image_files = []
    labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    image_files.append({
                        'image_file': class_name + "/" + image_name,
                        'label': class_name
                    })
                    labels.append(class_name)
    
    pageOn = 'dataset'
    return render_template('dataset.html', image_files=image_files, pageOn=pageOn)

@app.route("/training-testing/")
def trainingTesting():
    dataset_dir = "static/dataset"
    image_files = []
    labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    image_files.append(image_path)
                    labels.append(class_name)

    X_rgb = []
    X_hsv = []
    X_gray = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (32, 32))

        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        flattened_rgb = rgb_image.flatten()
        X_rgb.append(flattened_rgb)

        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        flattened_hsv = hsv_image.flatten()
        X_hsv.append(flattened_hsv)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        flattened_gray = gray_image.flatten()
        X_gray.append(flattened_gray)
    
    label_mapping = {label: i for i, label in enumerate(set(labels))}
    y = np.array([label_mapping[label] for label in labels])
    X_train_rgb, X_test_rgb, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)
    X_train_hsv, X_test_hsv, y_train, y_test = train_test_split(X_hsv, y, test_size=0.2, random_state=42)
    X_train_gray, X_test_gray, y_train, y_test = train_test_split(X_gray, y, test_size=0.2, random_state=42)

    X_train = np.concatenate((X_train_rgb, X_train_hsv, X_train_gray), axis=1)
    X_test = np.concatenate((X_test_rgb, X_test_hsv, X_test_gray), axis=1)

    k = 4
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    custom_label_mapping = {i: label for i, label in enumerate(label_mapping.keys())}
    knn_model_custom = {
        "model": knn,
        "label_mapping": custom_label_mapping
    }

    with open("static/save_model/knn_model.pkl", "wb") as f:
        pickle.dump(knn_model_custom, f)

    y_pred = knn.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('static/save_model/confusion_matrix.jpg', format='jpg')

    getAccuracy = accuracy*100
    pageOn = 'training-testing'

    return render_template('training-testing.html', pageOn=pageOn, getAccuracy=getAccuracy, classification_report=classification_rep)

dataset_dir = "static/dataset"
label_mapping = {}
with open("static/save_model/knn_model.pkl", "rb") as f:
    knn_model_custom = pickle.load(f)
    label_mapping = knn_model_custom["label_mapping"]
knn = knn_model_custom["model"]

def predict_label(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (32, 32))

    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).flatten()
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV).flatten()
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).flatten()

    feature_vector = np.concatenate((rgb_image, hsv_image, gray_image))

    label_index = knn.predict([feature_vector])[0]
    predicted_label = label_mapping[label_index]

    return predicted_label

@app.route("/testing-manually/", methods=['GET', 'POST'])
def testingManually():
    imageName = '?'
    labelName = '?'
    reccomendationName = '?'

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            labelName = predict_label(image_path)
            if (labelName == '2'):
                reccomendationName = '175kg/ha'
            elif (labelName == '3'):
                reccomendationName = '150kg/ha'
            elif (labelName == '4'):
                reccomendationName = '125kg/ha'
            elif (labelName == '5'):
                reccomendationName = '100kg/ha'

            imageName = filename

    displayImageName = imageName
    displaylabel = labelName
    displayReccomendation = reccomendationName
    pageOn = 'testing-manually'
    return render_template('testing-manually.html', pageOn=pageOn, displayImageName = displayImageName, displaylabel = displaylabel, displayReccomendation = displayReccomendation);

@app.route("/api-documentation/")
def apiDocumentation():
    pageOn = 'api-documentation'
    return render_template('api-documentation.html', pageOn=pageOn);
# End Route

# Start API
@app.route('/api/dataset/', methods=['GET'])
def get_dataset():
    dataset_dir = "./../dataset_nitrogen"
    image_files = []
    labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    image_files.append({
                        'image_file': class_name + "/" + image_name,
                        'label': class_name
                    })
                    labels.append(class_name)

    dataset = {
        'image_files': image_files,
        'labels': labels
    }

    return jsonify(dataset)

@app.route('/api/training-testing/', methods=['GET'])
def apiTrainingTesting():
    dataset_dir = "static/dataset"
    image_files = []
    labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    image_files.append(image_path)
                    labels.append(class_name)

    X_rgb = []
    X_hsv = []
    X_gray = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (32, 32))

        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        flattened_rgb = rgb_image.flatten()
        X_rgb.append(flattened_rgb)

        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        flattened_hsv = hsv_image.flatten()
        X_hsv.append(flattened_hsv)

        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        flattened_gray = gray_image.flatten()
        X_gray.append(flattened_gray)
    
    label_mapping = {label: i for i, label in enumerate(set(labels))}
    y = np.array([label_mapping[label] for label in labels])
    X_train_rgb, X_test_rgb, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)
    X_train_hsv, X_test_hsv, y_train, y_test = train_test_split(X_hsv, y, test_size=0.2, random_state=42)
    X_train_gray, X_test_gray, y_train, y_test = train_test_split(X_gray, y, test_size=0.2, random_state=42)

    X_train = np.concatenate((X_train_rgb, X_train_hsv, X_train_gray), axis=1)
    X_test = np.concatenate((X_test_rgb, X_test_hsv, X_test_gray), axis=1)

    k = 4
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    custom_label_mapping = {i: label for i, label in enumerate(label_mapping.keys())}
    knn_model_custom = {
        "model": knn,
        "label_mapping": custom_label_mapping
    }

    with open("static/save_model/knn_model.pkl", "wb") as f:
        pickle.dump(knn_model_custom, f)

    y_pred = knn.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('static/save_model/confusion_matrix.jpg', format='jpg')

    getAccuracy = accuracy*100

    response = {
        'classification_report_image' : 'classification_report.png',
        'confusion_matrix_image' : 'confusion_matrix.jpg',
        'getAccuracy': getAccuracy,
        'classification_report': classification_rep
    }

    return jsonify(response)

@app.route("/api/testing-manually/", methods=['POST'])
def apiTestingManually():
    label_mapping = {
        '2': '175kg/ha',
        '3': '150kg/ha',
        '4': '125kg/ha',
        '5': '100kg/ha'
    }

    response = {
        'imageName': '?',
        'labelName': '?',
        'recommendationName': '?'
    }

    if 'image' in request.files:
        file = request.files['image']
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            labelName = predict_label(image_path)
            response['imageName'] = filename
            response['labelName'] = labelName
            response['recommendationName'] = label_mapping.get(labelName, '?')

    return jsonify(response)
# End API

if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(debug=True, host='192.168.109.57')
    app.run(debug=True, host='192.168.18.209')
