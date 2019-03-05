import face_recognition
# 拉数据库，详细的在第一张报告里有解析。

# Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
# You can do that by using the face_distance function.
# 这个是做相似度检测的一个程序，不带识别功能，识别功能也是基于相似度的一个高低来判断的。

# The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
# be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
# positive matches at the risk of more false negatives.
# 但是对于亚洲人来说，0.6的相似度其实是远远不够的，经过一定的测试，相似的在0.4是相对符合亚洲人的人脸识别需求的。所以下面的容错我下调到0.4作为标准。

# Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
# smaller distance are more similar to each other than ones with a larger distance.

# Load some images to compare against 加载图片，作为数据库中的训练数据使用。
# 数据库中可以储存不止一张图片，请注意！！！
# 数据库中可以储存不止一张图片，请注意！！！
# 数据库中可以储存不止一张图片，请注意！！！
# 数据库中可以储存不止一张图片，请注意！！！

known_Justin_image = face_recognition.load_image_file("1.jpg")

# Get the face encodings for the known images 给认识的脸encodings。
Justin_face_encoding = face_recognition.face_encodings(known_Justin_image)[0]

known_encodings = [
    Justin_face_encoding
]

# Load a test image and get encondings for it 加载被检测图片，被检测图片也要赋予一个encoding。
image_to_test = face_recognition.load_image_file("2.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces。 做一个distance（相似度，解析度什么都行）的分析，在数据库中的照片和被检测照片之间对比。
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

# 输出distance并且判断是否相似
for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- 如果是同个人，辨析度应该小于0.4.那么这个人与数据库中的人是同个人吗？（T:是同个人； F：不是同个人） {}".format(face_distance < 0.4))
    print("- 如果是同个人，辨析度严格来说是应该小于0.3.那么这个人与数据库中的人是同个人吗？（T:是同个人； F：不是同个人） {}".format(face_distance < 0.3))
    print()
