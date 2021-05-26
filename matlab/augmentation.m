function [ aug_data, aug_labels ] = augmentation( data, num_key, seginfo )

    aug_data = {};
    aug_labels = {};

    for i=1:num_key

        label = [];
        for j=0:(num_key - 1)
            label = [label, i - 1, j];
        end

        for j=1:num_key
            k_range = j:num_key;

            for k=k_range
                for include_before=[true, false]
                    for include_end=[true, false]
                        if j == 1
                            if include_before
                                b = 1;
                            else
                                b = seginfo.peaks_b{i}(j);
                            end
                        else
                            if include_before
                                b = seginfo.peaks_e{i}(j - 1);
                            else
                                b = seginfo.peaks_b{i}(j);
                            end
                        end
                        if k == num_key
                            if include_end
                                e = numel(data(i).a.x);
                            else
                                e = seginfo.peaks_e{i}(k);
                            end
                        else
                            if include_end
                                e = seginfo.peaks_b{i}(k + 1);
                            else
                                e = seginfo.peaks_e{i}(k);
                            end
                        end
                        range = b:e;

                        for s='ag'
                            for axis='xyz'
                                sample.(s).(axis) = data(i).(s).(axis)(range);
                            end
                        end

                        aug_data = [aug_data, sample];
                        aug_labels = [aug_labels, label(((j - 1) * 2 + 1):(k * 2))];
                    end
                end
            end
        end
    end

end

