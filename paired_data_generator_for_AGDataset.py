import cv2 as cv
import numpy as np
import argparse
import glob
import os
import math
import time
import pandas as pd
import kornia



def rotate_image(mat, angle):


	height, width, _ = mat.shape # image shape has 3 dimensions
	image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

	rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])

	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)

	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]

	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	return rotated_mat


def rotate_image_same_size(mat, angle):


	height, width, _ = mat.shape # image shape has 3 dimensions
	image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

	rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv.warpAffine(mat, rotation_mat, (width, height))
	return rotated_mat


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)
   

    return img_crop, img_rot


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def salt_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def pepper_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

# parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! ')
# parser.add_argument('--input', help='Path to input image.', default='./')
# parser.add_argument('--output', help='Path to input image.', default='./hole_filling/')
# parser.add_argument('--cuda', help='whether to use CUDA.', default=0)
# args = parser.parse_args()
# inputdir = args.input
# outputdir = args.output



multi_image = True
add_mask = False
change_contrast = False
mask_type = "Gaussion"
label_list = []

if(multi_image):


	text_file = open("gt_manual.txt", "r")
	lines = text_file.readlines()
	total_image_number = len(lines)
	ortho_no_car = cv.imread("./pixel_1_rot_black.png")
	road_seg = cv.imread("./mask_rot_black_resize.jpg")

	for i in range(total_image_number):
		pic_name = lines[i].split()[0]
		tf_x = int(float(lines[i].split()[1]))
		tf_y = int(float(lines[i].split()[2]))
		print("TF", tf_x, tf_y)
		tf_rotation = float(lines[i].split()[3])
		if tf_rotation < 0:
			tf_rotation += 360.0
		# tf_scale = float(lines[i].split()[4])
		print("image number= ", pic_name)
		temp = cv.imread("./elevation/"+pic_name+".png")
		temp_resize = rotate_image_same_size(temp, tf_rotation)

		# temp_width = int(temp_r.shape[1]*tf_scale)
		# temp_hight = int(temp_r.shape[0]*tf_scale)
		temp_dim = (188, 188)
		temp_resize = cv.resize(temp_resize, temp_dim)
		print("image size = ", temp_resize.shape)

		for trans in range(10):
			#define the shift of the random_gt to temp_gt.
			shift_y_bottom = np.min([50, abs(tf_y - temp_resize.shape[0]/2)])
			shift_x_bottom = np.min([50, abs(tf_x - temp_resize.shape[1]/2)])
			shift_y_roof = 	 np.min([50, abs(ortho_no_car.shape[0]-tf_y-temp_resize.shape[0]/2)])
			shift_x_roof = 	 np.min([50, abs(ortho_no_car.shape[1]-tf_x-temp_resize.shape[1]/2)])
			shift_y = np.random.randint(-shift_y_bottom,shift_y_roof)
			shift_x = np.random.randint(-shift_x_bottom,shift_x_roof)


			shift_x , shift_y = 0,0
				# print("tf y = ", tf_y)
				# print("tf x = ", tf_x)
			print("shift y = ", shift_y)
			print("shift x = ", shift_x)
			print("tf_rot = ", tf_rotation)
			
			random_angle = np.random.randint(0,178)+np.random.rand()
			# image_gt_rotate = rotate_image_same_size(image_gt, random_angle)
			new_tf_rotation = random_angle
			rect = ((tf_x+temp_dim[0]/2,tf_y+temp_dim[1]/2),temp_dim, random_angle)
			print("Rect", rect)
			box = cv.boxPoints(rect)
			box = np.int0(box)
			# img_crop will the cropped rectangle, img_rot is the rotated image
			# cv.imshow("img_crop", ortho_no_car)
			# cv.waitKey(1)			
			img_crop_aerial, img_rot = crop_rect(ortho_no_car, rect)
			img_crop_road_seg, img_rot = crop_rect(road_seg, rect)		
			print("image—gt", img_crop_aerial.shape)	
			print("new rot =", new_tf_rotation)
			# shift_x_rotated = shift_x*np.cos(random_angle/180*np.pi) + shift_y*np.sin(random_angle/180*np.pi)
			# shift_y_rotated = shift_y*np.cos(random_angle/180*np.pi) - shift_x*np.sin(random_angle/180*np.pi)
			output_name = pic_name+"_"+str(-shift_x)+"_"+str(-shift_y)+"_"+str(new_tf_rotation)
			cv.imwrite('./data_train/ground/'+output_name+".jpg", temp_resize, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/aerial/'+output_name+".jpg", img_crop_aerial, [cv.IMWRITE_JPEG_QUALITY, 100])
			cv.imwrite('./data_train/road_seg/'+output_name+".jpg", img_crop_road_seg, [cv.IMWRITE_JPEG_QUALITY, 100])
			label_list.append({"name":output_name,"rotation":new_tf_rotation,"shift_x":-shift_x,"shift_y":-shift_y})
			# for rot in range(10):
			# 	random_rot = np.random.randint(0,90)
			# 	temp_r = rotate_image(temp, random_rot)
			# 	rotation_of_temp = tf_rotation+random_rot
			# 	print("image number= ", pic_name)
			# 	# print("tf y = ", tf_y)
			# 	# print("tf x = ", tf_x)
			# 	# print("shift y = ", shift_y)
			# 	# print("shift x = ", shift_x)
			# 	cv.imwrite('./data_train/ground/'+pic_name+"_"+str(-shift_x)+"_"+str(-shift_y)+"_"+str(rotation_of_temp)+".jpg", temp_r, [cv.IMWRITE_JPEG_QUALITY, 100])
			# 	cv.imwrite('./data_train/aerial/'+pic_name+"_"+str(-shift_x)+"_"+str(-shift_y)+"_"+str(rotation_of_temp)+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])


		# if(add_mask):
		# 	if(mask_type == "Gaussion"):
		# 		image_masked = cv.GaussianBlur(temp_resize,(5,5),0,0)
		# 	if(mask_type == "SaltPepper"):
		# 		image_masked = sp_noise(temp_resize, 0.01)
		# 	if(mask_type == "Salt"):
		# 		image_masked = salt_noise(temp_resize, 0.01)
		# 	if(mask_type == "Pepper"):
		# 		image_masked = pepper_noise(temp_resize, 0.01)

		# 	if(change_contrast):
		# 		image_c1 = cv.convertScaleAbs(image_masked, alpha=0.25, beta=0)
		# 		image_c2 = cv.convertScaleAbs(image_masked, alpha=0.5, beta=0)
		# 		image_c3 = cv.convertScaleAbs(image_masked, alpha=1.25, beta=0)
		# 		image_c4 = cv.convertScaleAbs(image_masked, alpha=1.5, beta=0)
		# 		# cv.imshow("image_c1",image_c1)
		# 		# cv.imshow("image_c2",image_c2)
		# 		# cv.imshow("image_c3",image_c3)
		# 		# cv.imshow("image_c4",image_c4)
		# 		# cv.imshow(mask_type,image_masked)
		# 		# cv.waitKey(5000)

		# 	# cv.imshow("ortho_"+pic_name, image_gt)
		# 	# cv.waitKey(5)
		# 	cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+".jpg", image_masked, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c1"+".jpg", image_c1, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c2"+".jpg", image_c2, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c3"+".jpg", image_c3, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/ground/'+pic_name+'_'+mask_type+"_c4"+".jpg", image_c4, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c1"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c2"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c3"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+"_c4"+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		# 	cv.imwrite('./data_train/aerial/'+pic_name+'_'+mask_type+".jpg", image_gt, [cv.IMWRITE_JPEG_QUALITY, 100])
		# else:
		# 	cv.imwrite('./data_train/ground/'+pic_name+"_c1"+".jpg", image_c1, [cv.IMWRITE_JPEG_QUALITY, 100])



else:
	text_file = open("tf.txt", "r")
	lines = text_file.readlines()
	total_image_number = len(lines)
	ortho_no_car = cv.imread("./image/ortho_no_car.jpg")

	for i in range(1):
		pic_name = lines[i].split()[0]
		tf_x = int(float(lines[i].split()[1]))
		tf_y = int(float(lines[i].split()[2]))
		tf_rotation = float(lines[i].split()[3])
		tf_scale = float(lines[i].split()[4])
		tf_x = 800
		tf_y =800
		tf_rotation = 92
		tf_scale = 2.49
		temp = cv.imread("./"+"48"+".jpg")
		temp_r = rotate_image(temp, -tf_rotation)
		temp_width = int(temp_r.shape[1]*tf_scale)
		temp_hight = int(temp_r.shape[0]*tf_scale)
		temp_dim = (temp_width, temp_hight)
		temp_resize = cv.resize(temp_r, temp_dim)

		cv.imwrite('./'+pic_name+".jpg", temp_resize, [cv.IMWRITE_JPEG_QUALITY, 100])

output_csv = pd.DataFrame(label_list, columns=["name","rotation","shift_x","shift_y"])
output_csv.to_csv("./ground_truth_qsdjt_lidar.csv",index=False)






