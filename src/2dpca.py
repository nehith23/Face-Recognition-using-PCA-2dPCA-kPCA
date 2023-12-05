from utils import get_faces, get_person_num, show_sample_faces, split_train_test, get_stats
import numpy as np
import timeit

faces = get_faces(zipfile_path="./Dataset.zip")

faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)
img_height, img_width = faceshape

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

training_set, testing_set = split_train_test(zipfilepath="./Dataset.zip")

start = timeit.default_timer()

def get_matrix(training_list, img_height, img_width):

    img_mat = np.zeros(
        (len(training_list), img_height, img_width),
        dtype=np.uint8)

    i = 0
    for img in training_list:
        mat = np.asmatrix(training_list[img])
        img_mat[i, :, :] = mat
        i += 1
    print("Matrix Size:", img_mat.shape)
    return img_mat



facematrix = get_matrix(training_set, img_height, img_width)
no_of_images = facematrix.shape[0]


mean_face = np.mean(facematrix, 0)

mean_subtracted = facematrix - mean_face


mat_width = facematrix.shape[2]
g_t = np.zeros((mat_width, mat_width))  # mxm

for i in range(no_of_images):

    temp = np.dot(mean_subtracted[i].T, mean_subtracted[i])
    g_t += temp

g_t /= no_of_images

eig_val, eig_vec = np.linalg.eig(g_t)

print("\nEnter the components: ",end="")
n =int(input())
eigfaces = eig_vec[:, 0:n]


weight_matrix = np.dot(facematrix, eigfaces)


def get_best_match(img):
    img_mat = testing_set[img]
    distances = []
    for i in range(no_of_images):
        temp_imgs = weight_matrix[i]
        dist = np.linalg.norm(img_mat@eigfaces - temp_imgs)
        distances += [dist]

    min = np.argmin(distances)
    return(min//8 + 1)

stop = timeit.default_timer()
correct_pred = 0
wrong_pred = 0
for img in testing_set:
    person_num, img_num = get_person_num(filename=img)

    best_match = get_best_match(img)
    if person_num == best_match:
        correct_pred += 1
    else:
        wrong_pred += 1
total_pred = correct_pred+wrong_pred

Accuracy = get_stats(correct_pred,wrong_pred,total_pred)
print(f"Correct prediction: ",correct_pred,"/",total_pred)
print(f"Wrong prediction: ",wrong_pred,"/",total_pred)

print(f"Accuracy: ",Accuracy,"%")
print(f"Time Taken: ",round(stop-start,3),"s")
