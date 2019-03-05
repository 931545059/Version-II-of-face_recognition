from PIL import Image
#引用一个PIL的库，拉一个image的函数
import face_recognition
#拉人脸识别算法

# Load the jpg file into a numpy array
#加载照片（在跟这个Python文件同个根目录下的一张照片，名字为1.jpg）
#上面的numpy array就是将图片转化到矩阵去方式，详细见学习笔记。用的是image这个函数。https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
image = face_recognition.load_image_file("1.jpg")

# Find all the faces in the image using the default HOG-based model.（这个HOG的全称，叫做Histogram of oriented gradients），中文：定向梯度的直方图。方向梯度直方图（HOG）是一个特征描述符中使用计算机视觉和图像处理为目的的对象检测。该技术计算图像的局部部分中的梯度方向的出现。该方法类似于边缘方向直方图，尺度不变特征变换描述符和形状上下文，但不同之处在于它是在均匀间隔的单元的密集网格上计算的，并且使用重叠的局部对比度归一化来提高精度。
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.这个方法很准确，但是没有cnn准确，cnn耗时长，也没有GPU加速，后面有算法是使用了GPU加速的，会处理的更加快。
# See also: find_faces_in_picture_cnn.py（这里就是提示，后面有个用CNN的方式寻找人脸，后补。耗时长但是准确度十分高）
#调用face recognition中的函数：face_locations就可以识别人脸的位置了
face_locations = face_recognition.face_locations(image)

#找到多少张脸，就会显示有多少张脸，提取这个数据使用{}来实现的。
print("I found {} face(s) in this photograph.".format(len(face_locations)))

#用一个for循环，调用第14行的取值face_locations，赋予people。

for people in face_locations:

    # Print the location of each face in this image。拿到图片信息，然后传到下面的print，这里的print是在输出结果中，（文字显示图片位置），不是实际截图的位置。）
    top, right, bottom, left = people
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    #在图片浏览器中拿到真实的数据，如下：
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
