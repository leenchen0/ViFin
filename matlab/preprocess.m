function [ processed_data ] = preprocess( data, labels, frame_size, frame_shift )

    % Segment
    for i=1:numel(data)
        for s='ag'
            for axis='xyz'
                signals.(s).(axis) = [];
                len = numel(data{i}.(s).(axis));
                for j=1:frame_shift:(len - frame_size + 1)
                    signals.(s).(axis) = [signals.(s).(axis); data{i}.(s).(axis)(j:(j + frame_size - 1))];
                end
            end
        end
        data{i} = signals;
    end

    % Processing
    for i=1:numel(data)
        concat = [];
        for s='ag'
            for axis='xyz'
                norm_data = zscore(data{i}.(s).(axis), 0, 2);
                concat = [concat, norm_data];
            end
        end
        data{i} = concat;
    end

    input_length = zeros(numel(data), 1, 'int64');
    label_length = zeros(numel(data), 1, 'int64');

    % get length of input and label
    for i=1:numel(data)
        input_length(i) = size(data{i}, 1);
        label_length(i) = numel(labels{i});
    end

    % get max length of input and label
    max_input_len = max(input_length);
    max_label_len = max(label_length);

    % padding labels and data
    labels_pad = zeros(numel(data), max_label_len, 'int64');
    for i=1:numel(data)
        label = labels{i};
        labels_pad(i, (1:numel(label))) = label;
    end

    data_pad = zeros(numel(data), max_input_len, size(data{1}, 2));
    for i=1:numel(data)
        data_pad(i, 1:size(data{i}, 1), :) = data{i};
    end

    processed_data = struct('data', data_pad, 'labels', labels_pad, 'input_length', input_length, 'label_length', label_length);

end

