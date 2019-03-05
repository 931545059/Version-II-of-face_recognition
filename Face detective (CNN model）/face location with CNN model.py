from PIL import Image
import face_recognition
#同之前的一样，拉取两个库，调用函数。

# Load the jpg file into a numpy array
#同之前一样，image的位置，导入图片。
image = face_recognition.load_image_file("1.jpg")

# Find all the faces in the image using a pre-trained convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# 用训练过的卷积神经网络convolutional neural network找人脸，也就是CNN的model。

# 速度快的版本请查看HOG-based的那个算法。
#其实代码都差不多，只是来运算的时候，model选择是CNN。
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for people in face_locations:

    # Print the location of each face in this image
    #获取图片资料和print图片信息与之前完全相符~不在多解释。详细在HOG-Based中。
    top, right, bottom, left = people
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

    #CNN这个算法对于“大合照”而言，效果欠佳。对于人物多的时候，CNN算法往往会漏算一些靠后的人，（甚至靠前的人也有漏算的情况）。但是CNN的优势在于可以准确的识别一个人，不会识别错误。算法相对耗时，精准度高但是识别率不高。
