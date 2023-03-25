dataset_folder_path = 'E:/college_project/dataset';

% Get a random image from the dataset
image = get_random_image(dataset_folder_path);

% Brightness correction (colored image)
brightened_image = imadjust(image,[.2 .3 0; .6 .7 1],[]);

% Contrast enhancement (colored image)
shadow_lab  = rgb2lab(brightened_image);
max_luminosity = 100;
L = shadow_lab(:,:,1)/max_luminosity;
shadow_adapthisteq = shadow_lab;
shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
shadow_adapthisteq = lab2rgb(shadow_adapthisteq);


subplot(1, 3, 1)
imshow(image), title("Original Image")

subplot(1, 3, 2)
imshow(shadow_adapthisteq), title("Brightness Correction")

subplot(1, 3, 3)
imshow(brightened_image), title("Contrast Enhancement")














% % Brightness Correction (grayscale image)
% bright = imadjust(gray_img);
% 
% subplot(2, 3, 6)
% imshow(bright), title("Brightness Correction")





% % Apply gaussian filter to the image
% gaussian_filtered_img = imgaussfilt3(image, 'FilterSize',3);
% 
% subplot(1, 3, 2)
% imshow(gaussian_filtered_img)

% bright = imadjust(image,[.2 .3 0; .6 .7 1],[]);
% subplot(1, 3, 3)
% imshow(bright), title("Brightness Correction")



% % Image enhancement (colored)
% shadow_lab  = rgb2lab(image);
% max_luminosity = 100;
% L = shadow_lab(:,:,1)/max_luminosity;
% 
% shadow_adapthisteq = shadow_lab;
% shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
% shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
% 
% 
% shadow_adapthisteq_gau = imgaussfilt3(shadow_adapthisteq, 'FilterSize',3);
% 
% 
% gray_img = rgb2gray(shadow_adapthisteq);
% J = adapthisteq(gray_img,'clipLimit',0.02,'Distribution','rayleigh');
% 
% % figure
% % montage({image, shadow_adapthisteq,J },"Size",[1 3])
% 
% subplot(1, 3, 1)
% imshow(image)
% 
% subplot(1,3, 2)
% imshow(shadow_adapthisteq)
% 
% subplot(1, 3, 3)
% imshow(shadow_adapthisteq_gau)
