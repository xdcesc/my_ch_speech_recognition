import tensorflow as tf

class DFSMN(object):
    def __init__(self, memory_size_left, memory_size_right, stride_l, stride_r,
                 input_size, output_size, dtype=tf.float32):
        self._memory_size_left = memory_size_left
        self._memory_size_right = memory_size_right
        self._memory_size = memory_size_left + memory_size_right + 1
        self._stride_l = stride_l
        self._stride_r = stride_r
        self._input_size = input_size
        self._output_size = output_size
        self._dtype = dtype
        self._build_graph()

    def _build_graph(self):
        self._W = tf.get_variable("dfsmnn_w", [self._input_size, self._output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=self._dtype))
        self._bias = tf.get_variable("dfsmnn_bias", [self._output_size],
                                    initializer=tf.constant_initializer(0.0, dtype=self._dtype))
        self._memory_weights =  tf.get_variable("memory_weights", [self._input_size, self._memory_size],
                                                initializer=tf.constant_initializer(1.0, dtype=self._dtype))

    def __call__(self, skip_con, input_data):
        size = input_data.get_shape()
        batch_size = size[0].value
        num_steps = size[1].value

        def skip_connection_func(input_skip):
            # Need to implement the transform
            return input_skip

        # Construct memory matrix
        memory_matrix = []
        for step in range(num_steps):
            left_num = tf.maximum(0, step - self._memory_size_left)
            right_num = tf.maximum(0, num_steps - step - self._memory_size_right - 1)
            weight_start = tf.minimum(-self._memory_size_left + step - 1, -1)
            weight_middle = -self._memory_size_left-1
            weight_end = tf.maximum(step - self._memory_size_left - num_steps, -self._memory_size)
            mem_l = self._memory_weights[weight_start:weight_middle:-1]
            mem_r = self._memory_weights[weight_middle-1:weight_end:-1]
            left_num = left_num - self._stride_l*len(mem_l)
            right_num = right_num - self._stride_r*len(mem_r)
            ele_1, ele_2 = 1, 0
            while ele_1 <= len(mem_l):
                for count1 in range(self._stride_l):
                    mem_l.insert(ele_1, self._input_size*[0])
                ele_1 += self._stride_l
            while ele_2 < len(mem_r):
                for count2 in range(self._stride_l):
                    mem_r.insert(ele_2, self._input_size*[0])
                ele_2 += self._stride_r

            mem = mem_l + self._memory_weights[self._memory_size_right] + mem_r
            mem = mem[tf.maximum(len(mem_l) - step, 0): tf.minimum(num_steps - step - 1 -len(mem_r), -1)]

            # strides padding
            if left_num <= 0:
                left_num = 0
            if right_num <=0:
                right_num = 0
            d_batch = tf.pad(mem, [[left_num, right_num], [0, 0]])
            memory_matrix.append([d_batch])
        memory_matrix = tf.concat(0, memory_matrix)

        # Compute the layer output
        h_hatt = tf.matmul([memory_matrix] * batch_size, input_data)
        p_s = skip_connection_func(skip_con)
        p_hatt = p_s + h_hatt
        h = tf.matmul(p_hatt, [self._W] * batch_size) + self._bias
        return h, p_hatt
