name: "MPIIGaze"
layers {
  name: "MPII_train"
  type: HDF5_DATA
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "../train_list.txt"
    batch_size: 1000
  }
  include: { phase: TRAIN }
}


layers {
  name: "MPII_test"
  type: HDF5_DATA
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "../test_list.txt"
    batch_size: 1000
  }
  include: { phase: TEST }
}

layers {
  name: "cutLabel"
  type: SLICE
  bottom: "label"
  top: "gaze"
  top: "headpose"
  slice_param {
    slice_dim: 1
    slice_point: 2
  }
}

layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "data"
  top: "conv1"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layers {
  name: "conv2"
  type: CONVOLUTION
  bottom: "pool1"
  top: "conv2"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}

layers {
  name: "pool2"
  type: POOLING
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}


##main net end##################################################################

name: "MPIIGazeLeft"
layers {
  name: "MPII_train"
  type: HDF5_DATA
  top: "dataleft"
  top: "labelleft"
  hdf5_data_param {
    source: "../train_list.txt"
    batch_size: 1000
  }
  include: { phase: TRAIN }
}


layers {
  name: "MPII_testLeft"
  type: HDF5_DATA
  top: "dataleft"
  top: "labelleft"
  hdf5_data_param {
    source: "../test_list.txt"
    batch_size: 1000
  }
  include: { phase: TEST }
}

layers {
  name: "cutLabel"
  type: SLICE
  bottom: "labelleft"
  top: "gazeleft"
  top: "headposeleft"
  slice_param {
    slice_dim: 1
    slice_point: 2
  }
}


layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "dataleft"
  top: "conv1left"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1left"
  top: "pool1left"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layers {
  name: "slice_left"
  type: SLICE
  bottom: "pool1left"
  top: "sliceleft1"
  top: "sliceleft2"
  slice_param{
    axis: 0
    slice_point: 30
    slice_point: 50
  }
}

layers {
  name: "concat_leftwithright"
  type: CONCAT
  bottom: "pool2left"
  bottom: "sliceright1"
  top: "poolLeftwithright"
  concat_param{
    axis: 0
  }
}

layers {
  name: "conv2"
  type: CONVOLUTION
  bottom: "poolLeftwithright"
  top: "conv2left"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}

layers {
  name: "pool2"
  type: POOLING
  bottom: "conv2left"
  top: "pool2left"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}



##left net end###################################################################

name: "MPIIGazeRight"
layers {
  name: "MPII_train"
  type: HDF5_DATA
  top: "dataright"
  top: "labelright"
  hdf5_data_param {
    source: "../train_list.txt"
    batch_size: 1000
  }
  include: { phase: TRAIN }
}


layers {
  name: "MPII_test"
  type: HDF5_DATA
  top: "dataright"
  top: "labelright"
  hdf5_data_param {
    source: "../test_list.txt"
    batch_size: 1000
  }
  include: { phase: TEST }
}

layers {
  name: "cutLabel"
  type: SLICE
  bottom: "labelright"
  top: "gazeright"
  top: "headposeright"
  slice_param {
    slice_dim: 1
    slice_point: 2
  }
}

layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "dataright"
  top: "conv1right"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1right"
  top: "pool1right"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}


layers {
  name: "slice_right"
  type: SLICE
  bottom: "pool1right"
  top: "sliceright1"
  top: "sliceright2"
  slice_param{
    axis: 0
    slice_point: 30
    slice_point: 50
  }
}

layers {
  name: "concat_rightwithleft"
  type: CONCAT
  bottom: "pool2right"
  bottom: "sliceleft1"
  top: "Rightwithleft"
  concat_param{
    axis: 0
  }
}

layers {
  name: "conv2"
  type: CONVOLUTION
  bottom: "Rightwithleft"
  top: "conv2right"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
}

layers {
  name: "pool2"
  type: POOLING
  bottom: "conv2right"
  top: "pool2right"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}


##right net end#################################################################

layers {
  name: "concat_all"
  type: CONCAT
  bottom: "pool2"
  bottom: "pool2left"
  bottom: "pool2right"
  top: "pool2all"
  concat_param{
    axis: 0
  }
}


layers {
  name: "ip1"
  type: INNER_PRODUCT
  bottom: "pool2all"
  top: "ip1"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}

layers {
  name: "relu1"
  type: RELU
  bottom: "ip1"
  top: "ip1"
}

layers {
  name: "concat_headpose_eyeappearance"
  type: CONCAT
  bottom: "ip1"
  bottom: "headpose"
  top: "cat"
}


layers {
  name: "ip2"
  type: INNER_PRODUCT
  bottom: "cat"
  top: "ip2"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "ip2"
  bottom: "gaze"
  top: "accuracy"
  include: { phase: TEST }
}
layers {
  name: "loss"
  type: EUCLIDEAN_LOSS
  bottom: "ip2"
  bottom: "gaze"
  top: "loss"
}