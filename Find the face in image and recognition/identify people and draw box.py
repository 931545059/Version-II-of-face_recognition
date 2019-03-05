import face_recognition
from PIL import Image, ImageDraw

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.
# 拉库和函数，不多解释了，看之前注解。

# Load a sample picture and learn how to recognize it.
# 加载第一张照片，并且赋予encoding值
# encoding值在最开始的project我写过
face1_image = face_recognition.load_image_file("E.jpg")
face1_face_encoding = face_recognition.face_encodings(face1_image)[0]

# Load a second sample picture and learn how to recognize it.
# 加载第一张照片，并且赋予encoding值
face2_image = face_recognition.load_image_file("J.jpg")
face2_face_encoding = face_recognition.face_encodings(face2_image)[0]

# Create arrays of known face encodings and their names
# 给认识的脸一个arrays，命名。
known_face_encodings = [
    face1_face_encoding,
    face2_face_encoding
]
known_face_names = [
    "Eason",
    "Jay"
]

# Load an image with an unknown face  加载检测照片
unknown_image = face_recognition.load_image_file("E&J1.jpg")

# Find all the faces and face encodings in the unknown image
# 用face_locations函数找出脸的位置，后面那个是图片位置，在31行已经赋值。
face_locations = face_recognition.face_locations(unknown_image)
# 给测试图片一个encoding值
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
# 官网解析需要看。
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # 这里这步，如果是在对比中发现没有这个的encoding，那么就会显示Unknown，在框框下。
    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a box around the face using the Pillow module
    # 输出结果
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255)) #蓝色框 这个就是接取出来的图相框。

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255)) #蓝色框 框在图像的什么位置，赋予一个定位
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255)) #白色字 以及字体应该在图像下的什么位置（并不是在框里，只是调整到显示在框里，字和框是分开的。）


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")
