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
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# all process
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
X_hist = []

for image_file in image_files:
    image = cv2.imread(image_file)
    resized_image = cv2.resize(image, (32, 32))

    mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    flattened_rgb = rgb_image.flatten()
    mean_rgb = np.mean(rgb_image, axis=(0, 1))
    mode_rgb = np.argmax(np.bincount(flattened_rgb))
    median_rgb = np.median(flattened_rgb)
    X_rgb.append(np.concatenate((flattened_rgb, mean_rgb, [mode_rgb], [median_rgb])))

    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    flattened_hsv = hsv_image.flatten()
    mean_hsv = np.mean(hsv_image, axis=(0, 1))
    mode_hsv = np.argmax(np.bincount(flattened_hsv))
    median_hsv = np.median(flattened_hsv)
    X_hsv.append(np.concatenate((flattened_hsv, mean_hsv, [mode_hsv], [median_hsv])))

    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    flattened_gray = gray_image.flatten()
    mean_gray = np.mean(gray_image)
    mode_gray = np.argmax(np.bincount(flattened_gray))
    median_gray = np.median(flattened_gray)
    X_gray.append(np.concatenate((flattened_gray, [mean_gray], [mode_gray], [median_gray])))

    hist = cv2.calcHist([masked_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_flattened = hist.flatten()
    X_hist.append(hist_flattened)

label_mapping = {label: i for i, label in enumerate(set(labels))}
y = np.array([label_mapping[label] for label in labels])

X_train_rgb, X_test_rgb, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)
X_train_hsv, X_test_hsv, y_train, y_test = train_test_split(X_hsv, y, test_size=0.2, random_state=42)
X_train_gray, X_test_gray, y_train, y_test = train_test_split(X_gray, y, test_size=0.2, random_state=42)
X_train_hist, X_test_hist, y_train, y_test = train_test_split(X_hist, y, test_size=0.2, random_state=42)

X_train = np.concatenate((X_train_rgb, X_train_hsv, X_train_gray, X_train_hist), axis=1)
X_test = np.concatenate((X_test_rgb, X_test_hsv, X_test_gray, X_test_hist), axis=1)

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

confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
display_cr = classification_report(y_test, y_pred, output_dict=True)
cr = classification_report(y_test, y_pred, output_dict=True)
df_classification_rep = pd.DataFrame(cr).transpose()

plt.figure(figsize=(10, 5))
sns.heatmap(df_classification_rep.iloc[:-1, :].T, annot=True, cmap='Blues', fmt=".2f")
plt.xlabel("Metrics")
plt.ylabel("Class")
plt.title("Classification Report")
plt.tight_layout()
plt.savefig("static/save_model/classification_report.png")

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Greens')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('static/save_model/confusion_matrix.jpg', format='jpg')
# end process 

# Start Route
@app.route("/")
def dashboard():
    pageOn = 'dashboard'
    akurasi = accuracy*100
    return render_template('dashboard.html', getAccuracy=akurasi, pageOn=pageOn)

@app.route("/dataset/")
def dataset():
    dataset_image_files = []
    dataset_labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    dataset_image_files.append({
                        'image_file': class_name + "/" + image_name,
                        'label': class_name
                    })
                    dataset_labels.append(class_name)
    
    pageOn = 'dataset'
    return render_template('dataset.html', image_files=dataset_image_files, pageOn=pageOn)

@app.route("/training-testing/")
def trainingTesting():
    getAccuracy = accuracy*100
    pageOn = 'training-testing'

    return render_template('training-testing.html', pageOn=pageOn, getAccuracy=getAccuracy, classification_report=display_cr)

def load_model():
    with open("static/save_model/knn_model.pkl", "rb") as f:
        knn_model_custom = pickle.load(f)
    return knn_model_custom

def classify_image(image_path, knn_model_custom):
    knn_model = knn_model_custom["model"]
    label_mapping = knn_model_custom["label_mapping"]

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (32, 32))

    mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    flattened_rgb = rgb_image.flatten()
    mean_rgb = np.mean(rgb_image, axis=(0, 1))
    mode_rgb = np.argmax(np.bincount(flattened_rgb))
    median_rgb = np.median(flattened_rgb)
    features_rgb = np.concatenate((flattened_rgb, mean_rgb, [mode_rgb], [median_rgb]))

    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    flattened_hsv = hsv_image.flatten()
    mean_hsv = np.mean(hsv_image, axis=(0, 1))
    mode_hsv = np.argmax(np.bincount(flattened_hsv))
    median_hsv = np.median(flattened_hsv)
    features_hsv = np.concatenate((flattened_hsv, mean_hsv, [mode_hsv], [median_hsv]))

    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    flattened_gray = gray_image.flatten()
    mean_gray = np.mean(gray_image)
    mode_gray = np.argmax(np.bincount(flattened_gray))
    median_gray = np.median(flattened_gray)
    features_gray = np.concatenate((flattened_gray, [mean_gray], [mode_gray], [median_gray]))

    hist = cv2.calcHist([masked_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_flattened = hist.flatten()

    features = np.concatenate((features_rgb, features_hsv, features_gray, hist_flattened))

    label_index = knn_model.predict([features])[0]
    predicted_label = label_mapping[label_index]

    return predicted_label

knn_model_custom = load_model()
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

            labelName = classify_image(image_path, knn_model_custom)
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
    dataset_image_files = []
    dataset_labels = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if image_path.endswith(".jpg"):
                    dataset_image_files.append({
                        'image_file': class_name + "/" + image_name,
                        'label': class_name
                    })
                    dataset_labels.append(class_name)

    dataset = {
        'image_files': image_files,
        'labels': labels
    }

    return jsonify(dataset)

@app.route('/api/training-testing/', methods=['GET'])
def apiTrainingTesting():
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

            labelName = classify_image(image_path, knn_model_custom)
            response['imageName'] = filename
            response['labelName'] = labelName
            response['recommendationName'] = label_mapping.get(labelName, '?')

    return jsonify(response)
# End API

if __name__ == '__main__':
    # app.run(debug=True) #server local
    app.run(debug=True, host='192.168.76.57') #server hotspot HP
    # app.run(debug=True, host='192.168.18.209') #server tobel
    # app.run(debug=True, host='192.168.18.209') #server was
