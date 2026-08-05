#chunk sizes
CHUNK_SIZE = 400


#data processing
PREPROCESS_LIST = ['scale','rotate_dim',['reduce_feat','pca']]
REDUCE_FEAT = 40
NN_AVG_MIN_ERROR = 0.001
MAX_NN_EPOCHS = 300

#boolean split settings
MAX_BOOL_SPLITS = 200

#maxtime predictor
MAX_TIME_PREDICTOR = 30

#further tree split crits
MIN_SAMPLES_IN_LEAF = 10 #greed_local generates new trees of depth 2 -> for new leafs, each leaf we would like to have 2 samples -> 10 is good last split point
MAX_GREEDY_DEPTH = 10
