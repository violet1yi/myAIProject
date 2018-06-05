#split the 20W images into samll files, each file has 5w images
import os.path as osp
def split_dataset():
    input_file = './anno/list_landmarks_align_celeba_strip.txt'
    output_file = open('part-list_landmarks_align_celeba.txt', 'w')

    count = 0
    with open(input_file) as input:
        for line in input:
            if count == 20002:
                break
                output_file.write(line)
                count += 1
    output_file.close()


# after resize the align images, their landmarks should be changed /4
def resize_landmarks(input_file, out_file):
    out = open(out_file, 'w')
    count = 0
    with open(input_file) as input:
        for line in input:
            line = line.strip().split()
            landmarks = line[1:]
            #print(landmarks)
            landmarks_resize = [int(int(i)/4) for i in landmarks]
            #print('land marks', landmarks_resize)
            landmarks_resize = [str(i) for i in landmarks_resize]
            string = " ".join(landmarks_resize)
            name = line[0].split('.')
            new_name = name[0] + '.png'
            new_line = new_name + ' ' + string + '\n'
            out.write(new_line)

    out.close()


def split_train_val(input, output1, output2):
    train_out = open(output1, 'w')
    val_out = open(output2, 'w')
    count = 0
    with open(input) as input:
        for line in input:
            if count < 20000:
                train_out.write(line)
            elif count < 25000:
                val_out.write(line)
            else:
                break
            count += 1
    train_out.close()
    val_out.close()



if __name__ == '__main__':
    #split_dataset()
    '''
    input_file = './anno/list_landmarks_align_celeba_strip.txt'
    output_file = './anno/resize-list_landmarks_align_celeba.txt'
    resize_landmarks(input_file, output_file)
    print('done')
    '''
    input = './anno/list_landmarks_celeba.txt'
    out1 = './anno/data/2w_train_list_landmarks_celeba.txt'
    out2 = './anno/data/5k_val_list_landmarks_celeba.txt'
    split_train_val(input, out1, out2)



