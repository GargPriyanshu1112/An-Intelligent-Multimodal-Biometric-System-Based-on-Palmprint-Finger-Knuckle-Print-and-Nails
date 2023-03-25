function [image] = get_random_image(dirpath)
% Get the list of images
file_list = dir(fullfile(dirpath, '/*.jpg'));

% Get a random image
rand_idx = randi(length(file_list), 1, 1);
rand_fpath = fullfile(dirpath, file_list(rand_idx).name);
image = imread(rand_fpath);

end

