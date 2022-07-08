import os


def create_images_list(datasets_path, save_path):
    for dataset in os.listdir(datasets_path):
        images_count = len(os.listdir(datasets_path + dataset + '/src'))
        f = open(save_path, 'a')
        for i in range(1, images_count+1):
            f.write(dataset + str(i) + '.png ' +
                    dataset + str(i) + '_mask.png' + '\n')
        print(
            f'Loaded dataset {dataset} with {images_count} images and masks.')
