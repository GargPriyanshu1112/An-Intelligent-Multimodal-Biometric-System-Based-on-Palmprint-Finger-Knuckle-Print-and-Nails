dataset_folder_path = 'E:/college_project/dataset';

% Get a random image from the dataset
image = get_random_image(dataset_folder_path);

% Gaussian smoothing (colored image)
smoothed_image_colored = imgaussfilt3(image, 'FilterSize',3);

% Contrast enhancement (colored image)
shadow_lab  = rgb2lab(smoothed_image_colored);
max_luminosity = 100;
L = shadow_lab(:,:,1)/max_luminosity;
shadow_adapthisteq = shadow_lab;
shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
shadow_adapthisteq = lab2rgb(shadow_adapthisteq);


% Grayscale the image
gray_img = rgb2gray(image);

% Gaussian smoothing (grayscale image)
smoothed_image_gray = imgaussfilt(gray_img, 'FilterSize',3);

% Contrast enhancement (grayscale image)
J = adapthisteq(smoothed_image_gray,'clipLimit',0.02,'Distribution','rayleigh');


subplot(2, 3, 1)
imshow(image), title("Original Image")

subplot(2, 3, 2)
imshow(smoothed_image_colored), title("Gaussian Smoothing")

subplot(2, 3, 3)
imshow(shadow_adapthisteq), title("Contrast Enhancement")

subplot(2, 3, 4)
imshow(gray_img), title("Grayscale Image")

subplot(2, 3, 5)
imshow(smoothed_image_gray), title("Gaussian Smoothing")

subplot(2, 3, 6)
imshow(J), title("Contrast Enhancement")