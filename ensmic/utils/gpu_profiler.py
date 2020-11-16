def get_max_memory_usage(sess):
    """Might be unprecise. Run after training"""
    if sess is None: sess = tf.get_default_session()
    max_mem = int(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
    print("Contrib max memory: %f G" % (max_mem / 1024. / 1024. / 1024.))
    return max_mem
