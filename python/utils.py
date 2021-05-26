
import numpy as np
from hdf5storage import loadmat as loadmat2
from tensorflow.keras.backend import ctc_decode, get_value

charset = {
    10: '0123456789',
    26: 'abcdefghijklmnopqrstuvwxyz'
}

def load_preprocessed_data(filename):
    mat = loadmat2(filename)
    data = mat['data']
    labels = mat['labels'].astype(int)
    input_length = mat['input_length']
    label_length = mat['label_length']
    num_key = mat['num_key'][0][0].astype(int)
    return data, labels, input_length, label_length, num_key

def edit_distance(target, source):
    len1 = len(target)
    len2 = len(source)
    dp = np.zeros((len1 + 1, len2 + 1))
    op = np.zeros((len1 + 1, len2 + 1), dtype=str)
    for i in range(len1 + 1):
        dp[i][0] = i
        op[i][0] = 'i' # Insert
    for j in range(len2 + 1):
        dp[0][j] = j
        op[0][j] = 'd' # Delete

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if target[i - 1] == source[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))

            if dp[i][j] == dp[i - 1][j - 1] + temp:
                op[i][j] = 'n' if temp == 0 else 'r'
            elif dp[i - 1][j] < dp[i][j - 1]:
                op[i][j] = 'i' # Insert
            else:
                op[i][j] = 'd' # Delete

    op_arr = []
    i = len1
    j = len2
    while (i > 0) or (j > 0):
        op_arr.insert(0, op[i][j])
        if op[i][j] == 'n' or op[i][j] == 'r':
            i -= 1
            j -= 1
        elif op[i][j] == 'i':
            i -= 1
        else:
            j -= 1

    return dp[len1][len2], op_arr

def print_accuracy(predicted, truth, num_key):
    count = [[0 for i in range(num_key + 2)] for i in range(num_key)]

    for i in range(len(predicted)):
        y_hat = predicted[i]
        y = truth[i]
        _, op = edit_distance(y, y_hat)
        y_pos = 0
        y_hat_pos = 0
        for j in range(len(op)):
            if op[j] != 'd':
                count[y[y_pos]][num_key + 1] += 1
            if op[j] == 'n':
                count[y[y_pos]][y[y_pos]] += 1
                y_pos += 1
                y_hat_pos += 1
            elif op[j] == 'r':
                count[y[y_pos]][y_hat[y_hat_pos]] += 1
                y_pos += 1
                y_hat_pos += 1
            elif op[j] == 'i':
                y_pos += 1
            else:
                count[y_hat[y_hat_pos]][num_key] += 1
                y_hat_pos += 1

    print('Count Table:')
    s = '     '
    for i in range(num_key):
        s += '%3d ' % i
    s += ' fa  all'
    print(s)
    for i in range(num_key):
        s = '%3d :' % i
        for j in range(num_key + 2):
            s += '%3d ' % count[i][j]
        print(s)

    print('Statistic:')
    print('label: all | right | wrong | miss | false alarm')
    count_sum = {'all': 0, 'right': 0, 'wrong': 0, 'miss': 0, 'fp': 0}
    for c in range(num_key):
        count_wrong = np.sum(count[c][:num_key]) - count[c][c]
        count_miss = count[c][num_key + 1] - count[c][c] - count_wrong
        print('%5d: %3d | %5d | %5d | %4d | %11d' % (c, count[c][num_key + 1], count[c][c], count_wrong, count_miss, count[c][num_key]))
        count_sum['all'] += count[c][num_key + 1]
        count_sum['right'] += count[c][c]
        count_sum['wrong'] += count_wrong
        count_sum['miss'] += count_miss
        count_sum['fp'] += count[c][num_key]
    print('all, [%d, %d, %d, %d, %d]' % (count_sum['all'], count_sum['right'], count_sum['wrong'], count_sum['miss'], count_sum['fp']))
    print('accuracy: %.2f%%' % (100.0 * count_sum['right'] / count_sum['all']))

    return count

def evaluate(model, data, labels, input_length, label_length, num_key):
    outputs = model.predict(data)
    y_pred = get_value(ctc_decode(outputs, np.squeeze(input_length, 1))[0][0])
    ed = 0
    all_y = []
    all_y_hat = []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        pred = pred[pred != -1]
        true = labels[i][:label_length[i][0]]
        y = label2char(true, charset[num_key])
        y_hat = label2char(pred, charset[num_key])
        y_hat_no_spell = y_hat
        all_y.append(true)
        all_y_hat.append(char2label(y_hat, charset[num_key]))
        print(y); print('==>'); print(y_hat)
        print()
        dis, _ = edit_distance(y, y_hat)
        ed += dis
    num = sum(label_length)
    print('%.2f%% (error = %d, all = %d)' % (100 - 100 * ed / num, ed, num))
    count = print_accuracy(all_y_hat, all_y, num_key)
    return count

def label2char(label, charset):
    char = []
    for l in label:
        char.append(charset[int(l)])
    return char

def char2label(char, charset):
    label = []
    for i in char:
        label.append(charset.find(i))
    return label
