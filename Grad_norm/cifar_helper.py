import tensorflow as tf


@tf.function()
def double_back(model_with_softmax, model, x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            p = model_with_softmax(x)
#             print(p)
            error = tf.losses.categorical_crossentropy(y, p)
        grad_Lx = tape2.gradient(error, x)
#         print(error)
        loss = error + lamb * tf.reduce_mean(tf.reduce_sum(grad_Lx ** 2, axis=[1,2,3]))
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_Lx

@tf.function()
def JacReg(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            p = model_with_softmax(x)
            error = tf.losses.categorical_crossentropy(y, p)
        grad_fx = tape2.batch_jacobian(p, x)
        loss = error + lamb * tf.reduce_sum(grad_fx ** 2) / tf.cast(tf.shape(y)[0], tf.float32)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_fx

@tf.function()
def FobReg(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            logit = model(x)
            error = tf.losses.categorical_crossentropy(y, tf.nn.softmax(logit))
        grad_gx = tape2.batch_jacobian(logit, x)
        loss = error + lamb * tf.reduce_sum(grad_gx ** 2) / tf.cast(tf.shape(y)[0], tf.float32)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx

@tf.function()
def MyFobReg2(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        # tr(g_x^T L^2g g_x)
        grad_gx = tape2.batch_jacobian(logit, x)
        grad_gx = tf.reshape(grad_gx, [-1,10,32*32*3])

        L_2g = tf.linalg.diag(p) - tf.matmul(tf.reshape(p,[-1,10,1]) , tf.reshape(p,[-1,1,10]))
        tmp = tf.matmul(tf.transpose(grad_gx, perm=[0,2,1]), L_2g)
        # To increase efficiency
        penalty = tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1])) / tf.cast(tf.shape(y)[0], tf.float32)


        loss = error + lamb * penalty
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx


@tf.function()
def MyFobReg1(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        # tr(g_x^T L^2g g_x)
        grad_gx = tape2.batch_jacobian(logit, x)
        grad_gx = tf.reshape(grad_gx, [-1,10,32*32*3])

        L_2g = tf.stop_gradient(tf.linalg.diag(p) - tf.matmul(tf.reshape(p,[-1,10,1]) , tf.reshape(p,[-1,1,10])))
        tmp = tf.matmul(tf.transpose(grad_gx, perm=[0,2,1]), L_2g)
        # To increase efficiency
        penalty = tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1])) / tf.cast(tf.shape(y)[0], tf.float32)


        loss = error + lamb * penalty
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx


@tf.function()
def dbMyFobReg1(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        grad_Lx = tape2.gradient(error, x)
        gLx_L2 = tf.reduce_mean(tf.reduce_sum(grad_Lx ** 2, axis=[1,2,3]))
        # tr(g_x^T L^2g g_x)
        grad_gx = tape2.batch_jacobian(logit, x)
        grad_gx = tf.reshape(grad_gx, [-1,10,32*32*3])

        L_2g = tf.stop_gradient(tf.linalg.diag(p) - tf.matmul(tf.reshape(p,[-1,10,1]) , tf.reshape(p,[-1,1,10])))
        tmp = tf.matmul(tf.transpose(grad_gx, perm=[0,2,1]), L_2g)
        # To increase efficiency
        penalty = tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1])) / tf.cast(tf.shape(y)[0], tf.float32)

        del tape2

        loss = error + lamb * (penalty + gLx_L2)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx


@tf.function()
def dbMyFobReg1_v2(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        grad_Lx = tape2.gradient(error, x)
        gLx_L2 = tf.reduce_mean(tf.reduce_sum(grad_Lx ** 2, axis=[1,2,3]))
        # tr(g_x^T L^2g g_x)
        grad_gx = tape2.batch_jacobian(logit, x)
        grad_gx = tf.reshape(grad_gx, [-1,10,32*32*3])

        L_2g = tf.stop_gradient(tf.linalg.diag(p) - tf.matmul(tf.reshape(p,[-1,10,1]) , tf.reshape(p,[-1,1,10])))
        tmp = tf.matmul(tf.transpose(grad_gx, perm=[0,2,1]), L_2g)
        # To increase efficiency
        penalty = tf.reduce_mean(tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1]),axis=[1,2]) ** 2)

        del tape2

        loss = error + lamb * (penalty + gLx_L2)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx


@tf.function()
def dbMyFobReg1_v3(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        grad_Lx = tape2.gradient(error, x)
        gLx_L2 = tf.reduce_mean(tf.reduce_sum(grad_Lx ** 2, axis=[1,2,3]))
        # tr(g_x^T L^2g g_x)
        grad_gx = tape2.batch_jacobian(logit, x)
        grad_gx = tf.reshape(grad_gx, [-1,10,32*32*3])

        L_2g = tf.stop_gradient(tf.linalg.diag(p) - tf.matmul(tf.reshape(p,[-1,10,1]) , tf.reshape(p,[-1,1,10])))
        tmp = tf.matmul(tf.matmul(tf.transpose(grad_gx, perm=[0,2,1]), L_2g), grad_gx)
        # To increase efficiency
        #print(tf.shape(tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1]),axis=[1,2]) ** 2))
        # tr(A) ^ 2 + 2 |A|_w^2
        penalty = tf.reduce_mean(tf.linalg.trace(tmp) ** 2) + 2 * tf.reduce_mean(tf.reduce_sum(tmp ** 2, axis=[1,2]))

        del tape2

        loss = error + lamb * (penalty + gLx_L2)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx


@tf.function()
def dbFobReg(model_with_softmax, model,x, y, lamb):
    with tf.GradientTape() as tape1:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            logit = model(x)
            p = tf.nn.softmax(logit)
            error = tf.losses.categorical_crossentropy(y, p)

        grad_Lx = tape2.gradient(error, x)
        #print(tf.shape(grad_Lx))
        gLx_L2 = tf.reduce_mean(tf.reduce_sum(grad_Lx ** 2, axis=[1,2,3]))
        # tr(g_x^T L^2g g_x)

        grad_gx = tape2.batch_jacobian(logit, x)
        penalty = tf.reduce_sum(grad_gx ** 2) / tf.cast(tf.shape(y)[0], tf.float32)

        # To increase efficiency
        #print(tf.shape(tf.reduce_sum(tmp * tf.transpose(grad_gx, perm=[0,2,1]),axis=[1,2]) ** 2))
        # tr(A) ^ 2 + 2 |A|_w^2

        del tape2


        loss = error + lamb * (penalty + gLx_L2)
    grads = tape1.gradient(loss, model.trainable_variables)

    return grads, loss, grad_gx
