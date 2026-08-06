#chunk sizes
CHUNK_SIZE = 200


#data processing
PREPROCESS_LIST = ['scale','rotate_dim',['reduce_feat','pca']]
REDUCE_FEAT = 25
NN_AVG_MIN_ERROR = 0.001
MAX_NN_EPOCHS = 500

#boolean split settings
AVG_BOOL_SPLITS_EACH_FEATURE = 10
MAX_SPLITS = 100

#maxtime predictor
MAX_TIME_PREDICTOR = 60

#further tree split crits
MIN_SAMPLES_IN_LEAF = 5 #greed_local generates new trees of depth 2 -> for new leafs, each leaf we would like to have 2 samples -> 8 is good last split point
MAX_GREEDY_DEPTH = 10
