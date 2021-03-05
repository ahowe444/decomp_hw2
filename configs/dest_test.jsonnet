{
  dataset_reader: {
    type: 'srl_reader',
    lazy: false,
    token_indexers: {
      words: {
        type: 'single_id'
      }
    }
  },
  train_data_path: 'data/decomp_data/dest_train.data',
  validation_data_path: 'data/decomp_data/dest_test.data',
  model: {
    type: 'srl_lstm',
    embedder: {
      token_embedders: {
        words: {
          type: 'embedding',
            pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
            embedding_dim: 50,
            trainable: false
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 10
    }
  },
  trainer: {
    num_epochs: 10,
    patience: 3,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.001
    }
  }
}
