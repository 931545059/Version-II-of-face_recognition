
import time
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
#拉库，注解在前面已标注

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#支持的格式设置 picture format


time_start=time.time()
#计时器开始 timer start

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    A = []
    b = []

    #查看在训练集里的所有人（文件夹） 
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 查看文件夹里的照片
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)  #root of image
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
        
                # Situation 1：No people in the picture, return: ound more than one face.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # If there has face in picture, 把这个人脸放去训练，跟之前的一样，赋予图片一个encoding值
                A.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                b.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(A))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(A, b)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(A_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    #英文不知道怎么解释，这个0.6就是我们所说的辨析度，辨析度高就是更加准确的辨别，但是辨析度太高了也很容易认不出来不同照片里的同个人。反之相反。
  
    if not os.path.isfile(A_img_path) or os.path.splitext(A_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(A_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model 
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    #  Find face in picture which I do before
    A_img = face_recognition.load_image_file(A_img_path)
    A_face_locations = face_recognition.face_locations(A_img)

    # If no faces are found in the picture, return an empty empty.
    if len(A_face_locations) == 0:
        return []

    # Find encodings(HOG-Based) for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(A_img, known_face_locations=A_face_locations)

    # Use the KNN model to find the best matches for the test face（Using the encoding value I did before）
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(A_face_locations))]

    #  Remove classifications that aren't within the threshold(set before in line 61)
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), A_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
   
    pil_image = Image.open(img_path).convert("RGB")
    #直观输出图片，直接open结果。
    #img_path is the root of the picture.
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
    #predictions is the result before.

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a box (same as before) write a name below.
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from harddrive. 
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")


    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)


time_end=time.time()
#timer stop
print('totally cost',time_end-time_start)
#output total time