import tensorflow as tf

x_train = [i for i in range(0, 20, 2)]  # even
x_val = [i for i in range(1, 20, 2)]  # odd
y_train = [i**2 for i in x_train]  # squared
y_val = [i**2 for i in x_val]

def gen_data_epoch(test=False):  # parametrized generator
    train_data = x_val if test else x_train
    label_data = y_val if test else y_train
    n_tests = len(train_data)
    for test_idx in range(len(train_data)):
        yield train_data[test_idx], label_data[test_idx]

def get_dataset(test=False):
    return tf.data.Dataset.from_generator(
        gen_data_epoch, args=(test,),
        output_types=(tf.int32, tf.int32))

print("Train:", [(i[0].numpy(), i[1].numpy()) for i in get_dataset().take(5)])
print("Test: ", [(i[0].numpy(), i[1].numpy()) for i in get_dataset(test=True).take(5)])

for x in gen_data_epoch():
    print(x)