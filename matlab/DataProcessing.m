
clear;

%% Config

data_folder = '../data/segmented';
saved_folder = '../data/processed';
num_round = 5;
fs = 100;
frame_size = int32(0.5 * fs);
frame_shift = int32(0.1 * fs);

%% Load Test data

mat = load(fullfile(data_folder, 'test.mat'), 'data', 'num_key', 'labels');
num_key = mat.num_key;
data_test = mat.data;
labels_test = mat.labels;


%% Data Augmentation

fprintf('Data Augmentation...\n');

data_train = cell(1, num_round);
labels_train = cell(1, num_round);

for round_no=1:num_round
    mat = load(fullfile(data_folder, sprintf('round%d.mat', round_no)), 'data', 'num_key', 'seginfo');
    [aug_data, aug_labels] = augmentation(mat.data, mat.num_key, mat.seginfo);
    data_train{round_no} = aug_data;
    labels_train{round_no} = aug_labels;
end


%% Data Preprocessing

fprintf('Data Preprocessing...\n');

processed_training_data = [];

for round_no=1:num_round
    processed_training_data = [processed_training_data, preprocess(data_train{round_no}, labels_train{round_no}, frame_size, frame_shift)];
end

processed_test_data = preprocess(data_test, labels_test, frame_size, frame_shift);


%% Merge Training Data

fprintf('Merging Training Data...\n');

feature_w = size(processed_training_data(1).data, 3);
time_len = zeros(1, num_round);
num_sample = zeros(1, num_round);
label_len = zeros(1, num_round);
for i=1:num_round
    time_len(i) = size(processed_training_data(i).data, 2);
    num_sample(i) = size(processed_training_data(i).data, 1);
    label_len(i) = size(processed_training_data(i).labels, 2);
end

% Merge data
data = zeros(sum(num_sample), max(time_len), feature_w);
count = 1;
for i=1:num_round
    data(count:(count + num_sample(i) - 1), 1:time_len(i), :) = processed_training_data(i).data;
    count = count + num_sample(i);
end

% Merge labels
labels = zeros(sum(num_sample), max(label_len));
count = 1;
for i=1:num_round
    labels(count:(count + num_sample(i) - 1), 1:label_len(i)) = processed_training_data(i).labels;
    count = count + num_sample(i);
end

% Merge input and label length
input_length = [];
label_length = [];
for i=1:num_round
    input_length = [input_length; processed_training_data(i).input_length];
    label_length = [label_length; processed_training_data(i).label_length];
end

processed_training_data = struct('data', data, 'labels', labels, 'input_length', input_length, 'label_length', label_length);


%% Save

fprintf('Saving...\n');

data = processed_training_data.data;
labels = processed_training_data.labels;
input_length = processed_training_data.input_length;
label_length = processed_training_data.label_length;
save(fullfile(saved_folder, 'training.mat'), 'data', 'labels', 'input_length', 'label_length', 'num_key', '-v7.3');

data = processed_test_data.data;
labels = processed_test_data.labels;
input_length = processed_test_data.input_length;
label_length = processed_test_data.label_length;
save(fullfile(saved_folder, 'test.mat'), 'data', 'labels', 'input_length', 'label_length', 'num_key', '-v7.3');

